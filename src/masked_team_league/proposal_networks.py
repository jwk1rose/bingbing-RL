from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import MatchFormat


@dataclass(frozen=True)
class ProposalNetworkConfig:
    loadout_count: int
    model_dim: int = 256
    heads: int = 8
    layers: int = 2
    max_slots: int = 25

    def __post_init__(self) -> None:
        if self.loadout_count <= 0:
            raise ValueError("loadout_count must be positive")
        if self.model_dim % self.heads != 0:
            raise ValueError("model_dim must be divisible by heads")


@dataclass(frozen=True)
class ProposalNetworkOutput:
    logits: Tensor
    masked_logits: Tensor
    value_estimates: Tensor
    gap_estimates: Tensor
    anti_meta_residual_estimates: Tensor


@dataclass(frozen=True)
class ProposalSequence:
    token_ids: tuple[int, ...]
    log_prob: float
    value_estimate: float
    gap_estimate: float
    anti_meta_residual_estimate: float = 0.0


@dataclass(frozen=True)
class GenerationContextOutput:
    context_memory: Tensor
    context_vector: Tensor


@dataclass(frozen=True)
class ProposalDistillationLoss:
    total_loss: Tensor
    behavior_cloning_loss: Tensor
    value_loss: Tensor
    gap_loss: Tensor
    anti_meta_residual_loss: Tensor


def build_causal_attention_mask(length: int, *, device: torch.device | str | None = None) -> Tensor:
    if length <= 0:
        raise ValueError("length must be positive")
    mask = torch.zeros((length, length), dtype=torch.float32, device=device)
    mask = mask.masked_fill(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask


def apply_legal_action_mask(logits: Tensor, legal_action_mask: Tensor) -> Tensor:
    if logits.shape != legal_action_mask.shape:
        raise ValueError("logits and legal_action_mask must have the same shape")
    return logits.masked_fill(~legal_action_mask.bool(), float("-inf"))


@torch.no_grad()
def beam_search_proposal_tokens(
    network: "_CausalPointerProposalNetwork",
    *,
    legal_action_mask_fn: Callable[[tuple[int, ...]], Sequence[bool] | Tensor],
    max_steps: int,
    beam_size: int,
    context_vector: Tensor | None = None,
    device: torch.device | str | None = None,
) -> tuple[ProposalSequence, ...]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    model_device = torch.device(device) if device is not None else next(network.parameters()).device
    network.eval()
    beams = (ProposalSequence(token_ids=(), log_prob=0.0, value_estimate=0.0, gap_estimate=0.0),)
    for _step in range(max_steps):
        next_beams: list[ProposalSequence] = []
        for beam in beams:
            legal = _legal_mask_tensor(legal_action_mask_fn(beam.token_ids), network.config.loadout_count, model_device)
            legal_count = int(legal.sum().item())
            if legal_count <= 0:
                continue
            selected = _selected_prefix_tensor(beam.token_ids, model_device)
            output = network(selected, legal.unsqueeze(0), context_vector=_context_for_single(context_vector, model_device))
            log_probs = torch.log_softmax(output.masked_logits, dim=-1).squeeze(0)
            top_k = min(beam_size, legal_count)
            values, indices = torch.topk(log_probs, k=top_k)
            for value, index in zip(values, indices):
                if not torch.isfinite(value):
                    continue
                next_beams.append(
                    ProposalSequence(
                        token_ids=beam.token_ids + (int(index.item()),),
                        log_prob=beam.log_prob + float(value.item()),
                        value_estimate=float(output.value_estimates.item()),
                        gap_estimate=float(output.gap_estimates.item()),
                        anti_meta_residual_estimate=float(output.anti_meta_residual_estimates.item()),
                    )
                )
        beams = tuple(sorted(next_beams, key=lambda item: item.log_prob, reverse=True)[:beam_size])
        if not beams:
            break
    return beams


@torch.no_grad()
def sample_proposal_tokens(
    network: "_CausalPointerProposalNetwork",
    *,
    legal_action_mask_fn: Callable[[tuple[int, ...]], Sequence[bool] | Tensor],
    max_steps: int,
    temperature: float = 1.0,
    context_vector: Tensor | None = None,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> ProposalSequence:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    model_device = torch.device(device) if device is not None else next(network.parameters()).device
    generator = torch.Generator(device=model_device)
    if seed is not None:
        generator.manual_seed(int(seed))
    network.eval()
    token_ids: tuple[int, ...] = ()
    log_prob = 0.0
    value_estimate = 0.0
    gap_estimate = 0.0
    for _step in range(max_steps):
        legal = _legal_mask_tensor(legal_action_mask_fn(token_ids), network.config.loadout_count, model_device)
        if int(legal.sum().item()) <= 0:
            raise ValueError("legal_action_mask_fn returned no legal actions")
        selected = _selected_prefix_tensor(token_ids, model_device)
        output = network(selected, legal.unsqueeze(0), context_vector=_context_for_single(context_vector, model_device))
        log_probs = torch.log_softmax(output.masked_logits / float(temperature), dim=-1).squeeze(0)
        probabilities = torch.softmax(output.masked_logits.squeeze(0) / float(temperature), dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
        token_id = int(sampled.item())
        token_ids = token_ids + (token_id,)
        log_prob += float(log_probs[token_id].item())
        value_estimate = float(output.value_estimates.item())
        gap_estimate = float(output.gap_estimates.item())
        anti_meta_residual_estimate = float(output.anti_meta_residual_estimates.item())
    return ProposalSequence(
        token_ids=token_ids,
        log_prob=log_prob,
        value_estimate=value_estimate,
        gap_estimate=gap_estimate,
        anti_meta_residual_estimate=anti_meta_residual_estimate,
    )


def _selected_prefix_tensor(prefix: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.tensor([(0,) + tuple(token_id + 1 for token_id in prefix)], dtype=torch.long, device=device)


def _legal_mask_tensor(mask: Sequence[bool] | Tensor, loadout_count: int, device: torch.device) -> Tensor:
    result = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if result.shape != (loadout_count,):
        raise ValueError("legal_action_mask_fn must return shape [loadout_count]")
    return result


def _context_for_single(context_vector: Tensor | None, device: torch.device) -> Tensor | None:
    if context_vector is None:
        return None
    if context_vector.ndim != 2 or context_vector.shape[0] != 1:
        raise ValueError("context_vector must have shape [1, model_dim]")
    return context_vector.to(device=device)


def proposal_distillation_loss(
    logits: Tensor,
    target_token_ids: Tensor,
    *,
    legal_action_mask: Tensor | None = None,
    candidate_weights: Tensor | None = None,
    value_estimates: Tensor | None = None,
    value_targets: Tensor | None = None,
    gap_estimates: Tensor | None = None,
    gap_targets: Tensor | None = None,
    anti_meta_residual_estimates: Tensor | None = None,
    anti_meta_residual_targets: Tensor | None = None,
    value_weight: float = 1.0,
    gap_weight: float = 1.0,
    anti_meta_residual_weight: float = 1.0,
) -> ProposalDistillationLoss:
    """Oracle-to-proposal distillation loss.

    `logits` is `[batch, sequence, loadout_count]` and `target_token_ids` is
    `[batch, sequence]`. Optional `candidate_weights` can be `[batch]` or
    `[batch, sequence]`; it represents the soft teacher weights derived from
    oracle candidate values.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, loadout_count]")
    if target_token_ids.shape != logits.shape[:2]:
        raise ValueError("target_token_ids must have shape [batch, sequence]")
    if target_token_ids.min().item() < 0 or target_token_ids.max().item() >= logits.shape[-1]:
        raise ValueError("target_token_ids contain an out-of-range loadout index")
    masked_logits = logits
    if legal_action_mask is not None:
        if legal_action_mask.shape != logits.shape:
            raise ValueError("legal_action_mask must have the same shape as logits")
        target_legal = legal_action_mask.gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
        if not bool(target_legal.all().item()):
            raise ValueError("teacher target is illegal under the supplied legal_action_mask")
        masked_logits = apply_legal_action_mask(logits, legal_action_mask)
    token_losses = F.cross_entropy(
        masked_logits.reshape(-1, logits.shape[-1]),
        target_token_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_token_ids).float()
    token_weights = _sequence_weights(token_losses, candidate_weights)
    behavior_cloning_loss = (token_losses * token_weights).sum() / token_weights.sum().clamp_min(1e-12)
    value_loss = (
        logits.new_tensor(0.0)
        if value_estimates is None and value_targets is None
        else _weighted_mse(value_estimates, value_targets, candidate_weights)
    )
    gap_loss = (
        logits.new_tensor(0.0)
        if gap_estimates is None and gap_targets is None
        else _weighted_mse(gap_estimates, gap_targets, candidate_weights)
    )
    anti_meta_residual_loss = (
        logits.new_tensor(0.0)
        if anti_meta_residual_estimates is None and anti_meta_residual_targets is None
        else _weighted_mse(anti_meta_residual_estimates, anti_meta_residual_targets, candidate_weights)
    )
    total_loss = (
        behavior_cloning_loss
        + float(value_weight) * value_loss
        + float(gap_weight) * gap_loss
        + float(anti_meta_residual_weight) * anti_meta_residual_loss
    )
    return ProposalDistillationLoss(
        total_loss=total_loss,
        behavior_cloning_loss=behavior_cloning_loss,
        value_loss=value_loss,
        gap_loss=gap_loss,
        anti_meta_residual_loss=anti_meta_residual_loss,
    )


def _sequence_weights(reference: Tensor, candidate_weights: Tensor | None) -> Tensor:
    if candidate_weights is None:
        return torch.ones_like(reference, dtype=torch.float32)
    weights = candidate_weights.to(device=reference.device, dtype=torch.float32)
    if weights.shape == reference.shape[:1]:
        return weights.unsqueeze(-1).expand_as(reference)
    if weights.shape == reference.shape:
        return weights
    raise ValueError("candidate_weights must have shape [batch] or [batch, sequence]")


def _weighted_mse(estimates: Tensor | None, targets: Tensor | None, candidate_weights: Tensor | None) -> Tensor:
    if estimates is None or targets is None:
        raise ValueError("estimates and targets must be provided together")
    if estimates.shape != targets.shape:
        raise ValueError("estimates and targets must have the same shape")
    losses = (estimates.float() - targets.to(device=estimates.device, dtype=torch.float32)).pow(2)
    if candidate_weights is None:
        return losses.mean()
    weights = candidate_weights.to(device=estimates.device, dtype=torch.float32)
    if weights.ndim > losses.ndim:
        raise ValueError("candidate_weights has too many dimensions")
    while weights.ndim < losses.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(losses)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class GenerationContextEncoder(nn.Module):
    """Non-causal context encoder for observation, belief, pool, and goal features."""

    def __init__(self, config: ProposalNetworkConfig, *, numeric_feature_dim: int = 0) -> None:
        super().__init__()
        self.config = config
        self.numeric_feature_dim = int(numeric_feature_dim)
        if self.numeric_feature_dim < 0:
            raise ValueError("numeric_feature_dim must be non-negative")
        self.token_embedding = nn.Embedding(config.loadout_count + 1, config.model_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_slots, config.model_dim)
        self.source_embedding = nn.Embedding(4, config.model_dim)
        self.hidden_slot_embedding = nn.Parameter(torch.zeros(config.model_dim))
        self.numeric_projection = (
            nn.Linear(self.numeric_feature_dim, config.model_dim) if self.numeric_feature_dim > 0 else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)

    def forward(
        self,
        observation_token_ids: Tensor,
        *,
        observation_hidden_mask: Tensor | None = None,
        belief_token_ids: Tensor | None = None,
        belief_weights: Tensor | None = None,
        pool_token_ids: Tensor | None = None,
        numeric_features: Tensor | None = None,
    ) -> GenerationContextOutput:
        if observation_token_ids.ndim != 2:
            raise ValueError("observation_token_ids must have shape [batch, slots]")
        pieces = [
            self._embed_token_sequence(
                observation_token_ids,
                source_id=0,
                hidden_mask=observation_hidden_mask,
            )
        ]
        if belief_token_ids is not None:
            pieces.append(self._embed_belief_tokens(belief_token_ids, belief_weights))
        if pool_token_ids is not None:
            pieces.append(self._embed_token_sequence(pool_token_ids, source_id=2))
        if numeric_features is not None:
            if self.numeric_projection is None:
                raise ValueError("numeric_feature_dim must be positive when numeric_features are supplied")
            if numeric_features.ndim != 2 or numeric_features.shape[1] != self.numeric_feature_dim:
                raise ValueError("numeric_features must have shape [batch, numeric_feature_dim]")
            pieces.append(self.numeric_projection(numeric_features.float()).unsqueeze(1) + self.source_embedding.weight[3])
        memory = torch.cat(pieces, dim=1)
        encoded = self.encoder(memory)
        return GenerationContextOutput(context_memory=encoded, context_vector=encoded.mean(dim=1))

    def _embed_token_sequence(self, token_ids: Tensor, *, source_id: int, hidden_mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, slots]")
        batch_size, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device).clamp(max=self.config.max_slots - 1)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        embedded = self.token_embedding(token_ids.clamp(min=0, max=self.config.loadout_count))
        embedded = embedded + self.position_embedding(positions)
        embedded = embedded + self.source_embedding(torch.full_like(token_ids, int(source_id)))
        if hidden_mask is not None:
            if hidden_mask.shape != token_ids.shape:
                raise ValueError("hidden_mask must have the same shape as token_ids")
            embedded = embedded + hidden_mask.bool().unsqueeze(-1).to(embedded.dtype) * self.hidden_slot_embedding
        return embedded

    def _embed_belief_tokens(self, belief_token_ids: Tensor, belief_weights: Tensor | None) -> Tensor:
        if belief_token_ids.ndim != 3:
            raise ValueError("belief_token_ids must have shape [batch, candidates, slots]")
        batch_size, candidate_count, slot_count = belief_token_ids.shape
        flattened = belief_token_ids.reshape(batch_size, candidate_count * slot_count)
        embedded = self._embed_token_sequence(flattened, source_id=1)
        if belief_weights is None:
            return embedded
        if belief_weights.shape != (batch_size, candidate_count):
            raise ValueError("belief_weights must have shape [batch, candidates]")
        weights = belief_weights.to(device=embedded.device, dtype=embedded.dtype).unsqueeze(-1).expand(-1, -1, slot_count)
        return embedded * weights.reshape(batch_size, candidate_count * slot_count).unsqueeze(-1)


class _CausalPointerProposalNetwork(nn.Module):
    def __init__(self, config: ProposalNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.loadout_count + 1, config.model_dim, padding_idx=0)
        self.slot_embedding = nn.Embedding(config.max_slots, config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.pointer_head = nn.Linear(config.model_dim, config.loadout_count)
        self.value_head = nn.Linear(config.model_dim, 1)
        self.gap_head = nn.Linear(config.model_dim, 1)
        self.anti_meta_residual_head = nn.Linear(config.model_dim, 1)

    def forward(
        self,
        selected_token_ids: Tensor,
        legal_action_mask: Tensor,
        *,
        context_vector: Tensor | None = None,
    ) -> ProposalNetworkOutput:
        if selected_token_ids.ndim != 2:
            raise ValueError("selected_token_ids must have shape [batch, sequence]")
        if legal_action_mask.ndim != 2 or legal_action_mask.shape[1] != self.config.loadout_count:
            raise ValueError("legal_action_mask must have shape [batch, loadout_count]")
        batch_size, length = selected_token_ids.shape
        if length > self.config.max_slots:
            raise ValueError("selected sequence length exceeds max_slots")
        slot_ids = torch.arange(length, device=selected_token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(selected_token_ids.clamp(min=0, max=self.config.loadout_count))
        x = x + self.slot_embedding(slot_ids)
        causal_mask = build_causal_attention_mask(length, device=selected_token_ids.device)
        decoded = self.decoder(x, mask=causal_mask)
        query = decoded[:, -1, :]
        if context_vector is not None:
            if context_vector.shape != query.shape:
                raise ValueError("context_vector must have shape [batch, model_dim]")
            query = query + context_vector.to(device=query.device, dtype=query.dtype)
        logits = self.pointer_head(query)
        masked_logits = apply_legal_action_mask(logits, legal_action_mask)
        value_estimates = self.value_head(query).squeeze(-1)
        gap_estimates = self.gap_head(query).squeeze(-1)
        anti_meta_residual_estimates = self.anti_meta_residual_head(query).squeeze(-1)
        return ProposalNetworkOutput(
            logits=logits,
            masked_logits=masked_logits,
            value_estimates=value_estimates,
            gap_estimates=gap_estimates,
            anti_meta_residual_estimates=anti_meta_residual_estimates,
        )


class AttackGenerationNetwork(_CausalPointerProposalNetwork):
    """Causal proposal network for attack loadout sequences.

    The network produces logits over the loadout pool; the caller supplies the
    structural legal action mask from `ConstraintEngine`.
    """


class DefenseRosterGenerationNetwork(_CausalPointerProposalNetwork):
    """Causal proposal network for defense roster sequences."""


class MaskSelectionNetwork(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int = 128, leakage_penalty: float = 1.0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.slot_scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.leakage_penalty = float(leakage_penalty)

    def forward(self, slot_features: Tensor) -> Tensor:
        if slot_features.ndim != 3:
            raise ValueError("slot_features must have shape [batch, slots, feature_dim]")
        return self.slot_scorer(slot_features).squeeze(-1)

    def select_mask(
        self,
        slot_scores: Tensor,
        match_format: MatchFormat,
        *,
        position_leakage: Tensor | None = None,
        equip_leakage: Tensor | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        if slot_scores.shape != (match_format.n_teams, match_format.team_size):
            raise ValueError("slot_scores shape must match the match format")
        adjusted = slot_scores.detach().clone().float()
        if position_leakage is not None:
            adjusted = adjusted - self.leakage_penalty * position_leakage.detach().float()
        if equip_leakage is not None:
            adjusted = adjusted - self.leakage_penalty * equip_leakage.detach().float()
        chosen = [[0 for _slot in range(match_format.team_size)] for _team in range(match_format.n_teams)]
        per_team = [0 for _team in range(match_format.n_teams)]
        total = 0
        flat: list[tuple[float, int, int]] = []
        for team_idx in range(match_format.n_teams):
            for slot_idx in range(match_format.team_size):
                flat.append((float(adjusted[team_idx, slot_idx]), team_idx, slot_idx))
        flat.sort(key=lambda item: item[0], reverse=True)
        for _score, team_idx, slot_idx in flat:
            if total >= match_format.max_hidden_total:
                break
            if per_team[team_idx] >= match_format.max_hidden_per_team:
                continue
            chosen[team_idx][slot_idx] = 1
            per_team[team_idx] += 1
            total += 1
        return tuple(tuple(row) for row in chosen)
