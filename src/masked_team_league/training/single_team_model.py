from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..domain import Loadout, Team
from ..scoring import SurrogatePrediction, SurrogateScorer


@dataclass(frozen=True)
class LoadoutVocab:
    hero_to_index: Mapping[int, int]
    equip_to_index: Mapping[int | None, int]
    bucket_to_index: Mapping[str, int]

    @classmethod
    def from_loadouts(cls, loadouts: tuple[Loadout, ...]) -> "LoadoutVocab":
        hero_ids = sorted({loadout.hero_id for loadout in loadouts})
        equip_ids = sorted({loadout.unique_equip_id for loadout in loadouts if loadout.unique_equip_id is not None})
        buckets = sorted({loadout.standing_bucket for loadout in loadouts})
        return cls(
            hero_to_index={hero_id: index + 1 for index, hero_id in enumerate(hero_ids)},
            equip_to_index={None: 0, **{equip_id: index + 1 for index, equip_id in enumerate(equip_ids)}},
            bucket_to_index={bucket: index + 1 for index, bucket in enumerate(buckets)},
        )

    @property
    def hero_count(self) -> int:
        return max(self.hero_to_index.values(), default=0) + 1

    @property
    def equip_count(self) -> int:
        return max(self.equip_to_index.values(), default=0) + 1

    @property
    def bucket_count(self) -> int:
        return max(self.bucket_to_index.values(), default=0) + 1

    def hero_index(self, hero_id: int) -> int:
        return self.hero_to_index[int(hero_id)]

    def equip_index(self, equip_id: int | None) -> int:
        return self.equip_to_index.get(equip_id, 0)

    def bucket_index(self, bucket: str) -> int:
        return self.bucket_to_index.get(bucket, 0)

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "hero_to_index": {str(key): int(value) for key, value in self.hero_to_index.items()},
            "equip_to_index": {"none" if key is None else str(key): int(value) for key, value in self.equip_to_index.items()},
            "bucket_to_index": {str(key): int(value) for key, value in self.bucket_to_index.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Mapping[str, int]]) -> "LoadoutVocab":
        equip_mapping: dict[int | None, int] = {}
        for key, value in payload["equip_to_index"].items():
            equip_mapping[None if key == "none" else int(key)] = int(value)
        return cls(
            hero_to_index={int(key): int(value) for key, value in payload["hero_to_index"].items()},
            equip_to_index=equip_mapping,
            bucket_to_index={str(key): int(value) for key, value in payload["bucket_to_index"].items()},
        )


@dataclass(frozen=True)
class SingleTeamWinrateModelConfig:
    hero_dim: int = 128
    equip_dim: int = 32
    star_dim: int = 16
    bucket_dim: int = 16
    stats_dim: int = 64
    model_dim: int = 256
    pair_dim: int | None = None
    heads: int = 8
    layers: int = 2
    team_size: int = 5

    def __post_init__(self) -> None:
        if self.model_dim % self.heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        if self.team_size != 5:
            raise ValueError("SingleTeamWinrateModel currently expects team_size=5")


def encode_team_batch(
    attacks: tuple[Team, ...],
    defenses: tuple[Team, ...],
    vocab: LoadoutVocab,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    if len(attacks) != len(defenses):
        raise ValueError("attack and defense batch sizes must match")
    batch_size = len(attacks)
    hero_ids = torch.zeros((batch_size, 2, 5), dtype=torch.long, device=device)
    equip_ids = torch.zeros((batch_size, 2, 5), dtype=torch.long, device=device)
    star_ids = torch.zeros((batch_size, 2, 5), dtype=torch.long, device=device)
    bucket_ids = torch.zeros((batch_size, 2, 5), dtype=torch.long, device=device)
    features = torch.zeros((batch_size, 2, 5, 4), dtype=torch.float32, device=device)
    standing_ranks = torch.zeros((batch_size, 2, 5), dtype=torch.float32, device=device)
    star_values = torch.zeros((batch_size, 2, 5), dtype=torch.float32, device=device)
    powers = torch.zeros((batch_size, 2, 5), dtype=torch.float32, device=device)

    for batch_index, (attack, defense) in enumerate(zip(attacks, defenses)):
        for side_index, team in enumerate((attack, defense)):
            for slot_index, loadout in enumerate(team.slots):
                hero_ids[batch_index, side_index, slot_index] = vocab.hero_index(loadout.hero_id)
                equip_ids[batch_index, side_index, slot_index] = vocab.equip_index(loadout.unique_equip_id)
                star_id = 0 if loadout.unique_equip_star is None else int(loadout.unique_equip_star) - 2
                star_ids[batch_index, side_index, slot_index] = star_id
                bucket_ids[batch_index, side_index, slot_index] = vocab.bucket_index(loadout.standing_bucket)
                normal_count = float(len(loadout.normal_equip_ids))
                features[batch_index, side_index, slot_index] = torch.tensor(
                    [
                        loadout.final_power / 10000.0,
                        loadout.standing_rank / 1000.0,
                        loadout.cost / 10000.0,
                        normal_count / 10.0,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                standing_ranks[batch_index, side_index, slot_index] = float(loadout.standing_rank)
                star_values[batch_index, side_index, slot_index] = float(loadout.unique_equip_star or 0)
                powers[batch_index, side_index, slot_index] = float(loadout.final_power)
    return {
        "hero_ids": hero_ids,
        "equip_ids": equip_ids,
        "star_ids": star_ids,
        "bucket_ids": bucket_ids,
        "features": features,
        "standing_ranks": standing_ranks,
        "star_values": star_values,
        "powers": powers,
    }


class LoadoutEncoder(nn.Module):
    def __init__(self, vocab: LoadoutVocab, config: SingleTeamWinrateModelConfig) -> None:
        super().__init__()
        self.hero = nn.Embedding(vocab.hero_count, config.hero_dim, padding_idx=0)
        self.equip = nn.Embedding(vocab.equip_count, config.equip_dim, padding_idx=0)
        self.star = nn.Embedding(4, config.star_dim, padding_idx=0)
        self.bucket = nn.Embedding(vocab.bucket_count, config.bucket_dim, padding_idx=0)
        self.stats = nn.Sequential(
            nn.Linear(4, config.stats_dim),
            nn.GELU(),
            nn.Linear(config.stats_dim, config.stats_dim),
            nn.GELU(),
        )
        input_dim = config.hero_dim + config.equip_dim + config.star_dim + config.bucket_dim + config.stats_dim
        self.project = nn.Sequential(nn.Linear(input_dim, config.model_dim), nn.GELU(), nn.LayerNorm(config.model_dim))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        pieces = [
            self.hero(batch["hero_ids"]),
            self.equip(batch["equip_ids"]),
            self.star(batch["star_ids"]),
            self.bucket(batch["bucket_ids"]),
            self.stats(batch["features"]),
        ]
        return self.project(torch.cat(pieces, dim=-1))


class RelativeSelfAttentionBlock(nn.Module):
    def __init__(self, config: SingleTeamWinrateModelConfig) -> None:
        super().__init__()
        self.model_dim = config.model_dim
        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.qkv = nn.Linear(config.model_dim, config.model_dim * 3)
        self.out = nn.Linear(config.model_dim, config.model_dim)
        self.rel_slot = nn.Embedding(config.team_size * 2 - 1, config.heads)
        self.rel_rank = nn.Sequential(nn.Linear(2, config.heads), nn.Tanh())
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.GELU(),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )

    def forward(self, x: Tensor, ranks: Tensor) -> Tensor:
        batch_size, length, _dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch_size, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        slot_index = torch.arange(length, device=x.device)
        rel_index = slot_index[:, None] - slot_index[None, :] + length - 1
        slot_bias = self.rel_slot(rel_index).permute(2, 0, 1).unsqueeze(0)
        rank_delta = ranks[:, :, None] - ranks[:, None, :]
        rank_features = torch.stack((rank_delta / 1000.0, rank_delta.abs() / 1000.0), dim=-1)
        rank_bias = self.rel_rank(rank_features).permute(0, 3, 1, 2)
        attention = torch.softmax(logits + slot_bias + rank_bias, dim=-1)
        attended = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, length, self.model_dim)
        x = x + self.out(attended)
        x = x + self.ffn(self.norm2(x))
        return x


class OrderedTeamEncoder(nn.Module):
    def __init__(self, config: SingleTeamWinrateModelConfig) -> None:
        super().__init__()
        self.slot = nn.Embedding(config.team_size, config.model_dim)
        self.rank_slot = nn.Sequential(nn.Linear(2, config.model_dim), nn.GELU(), nn.Linear(config.model_dim, config.model_dim))
        self.blocks = nn.ModuleList(RelativeSelfAttentionBlock(config) for _ in range(config.layers))

    def forward(self, loadout_vectors: Tensor, ranks: Tensor) -> Tensor:
        batch_size, length, _dim = loadout_vectors.shape
        slot_index = torch.arange(length, device=loadout_vectors.device)
        slot_features = torch.stack(
            (
                ranks / 1000.0,
                slot_index.float().unsqueeze(0).expand(batch_size, -1) / max(length - 1, 1),
            ),
            dim=-1,
        )
        x = loadout_vectors + self.slot(slot_index).unsqueeze(0) + self.rank_slot(slot_features)
        for block in self.blocks:
            x = block(x, ranks)
        return torch.cat((x.mean(dim=1), x.max(dim=1).values), dim=-1)


class PairwiseInteraction(nn.Module):
    def __init__(self, config: SingleTeamWinrateModelConfig) -> None:
        super().__init__()
        pair_dim = config.pair_dim or config.model_dim
        self.pair_dim = pair_dim
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim * 4 + 4, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
        )
        self.attention = nn.Linear(pair_dim, 1)

    def forward(self, attack: Tensor, defense: Tensor, batch: dict[str, Tensor]) -> Tensor:
        attack_pair = attack[:, :, None, :].expand(-1, -1, 5, -1)
        defense_pair = defense[:, None, :, :].expand(-1, 5, -1, -1)
        attack_rank = batch["standing_ranks"][:, 0, :, None].expand(-1, -1, 5)
        defense_rank = batch["standing_ranks"][:, 1, None, :].expand(-1, 5, -1)
        attack_star = batch["star_values"][:, 0, :, None].expand(-1, -1, 5)
        defense_star = batch["star_values"][:, 1, None, :].expand(-1, 5, -1)
        attack_power = batch["powers"][:, 0, :, None].expand(-1, -1, 5)
        defense_power = batch["powers"][:, 1, None, :].expand(-1, 5, -1)
        delta_rank = (attack_rank - defense_rank) / 1000.0
        delta_star = (attack_star - defense_star) / 5.0
        delta_power = (attack_power - defense_power) / 10000.0
        pair_features = torch.stack((delta_rank, delta_rank.abs(), delta_star, delta_power), dim=-1)
        pair = torch.cat((attack_pair, defense_pair, attack_pair * defense_pair, attack_pair - defense_pair, pair_features), dim=-1)
        pair = self.mlp(pair.reshape(pair.shape[0], 25, -1))
        weights = torch.softmax(self.attention(pair).squeeze(-1), dim=-1)
        attended = torch.sum(pair * weights.unsqueeze(-1), dim=1)
        return torch.cat((pair.mean(dim=1), pair.max(dim=1).values, attended), dim=-1)


class SingleTeamWinrateModel(nn.Module):
    def __init__(self, vocab: LoadoutVocab, config: SingleTeamWinrateModelConfig | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = config or SingleTeamWinrateModelConfig()
        pair_dim = self.config.pair_dim or self.config.model_dim
        self.loadout_encoder = LoadoutEncoder(vocab, self.config)
        self.team_encoder = OrderedTeamEncoder(self.config)
        self.cross = PairwiseInteraction(self.config)
        team_dim = self.config.model_dim * 2
        final_dim = team_dim * 4 + pair_dim * 3 + 2
        self.trunk = nn.Sequential(
            nn.Linear(final_dim, self.config.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.model_dim),
            nn.Linear(self.config.model_dim, self.config.model_dim // 2),
            nn.GELU(),
        )
        hidden = self.config.model_dim // 2
        self.win_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)
        self.margin_head = nn.Linear(hidden, 1)
        self.duration_head = nn.Linear(hidden, 1)
        self.residual_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        encoded = self.loadout_encoder(batch)
        attack = encoded[:, 0]
        defense = encoded[:, 1]
        attack_ranks = batch["standing_ranks"][:, 0]
        defense_ranks = batch["standing_ranks"][:, 1]
        z_attack = self.team_encoder(attack, attack_ranks)
        z_defense = self.team_encoder(defense, defense_ranks)
        z_cross = self.cross(attack, defense, batch)
        delta_power = (batch["powers"][:, 0].sum(dim=1) - batch["powers"][:, 1].sum(dim=1)).unsqueeze(-1) / 10000.0
        delta_star = (batch["star_values"][:, 0].mean(dim=1) - batch["star_values"][:, 1].mean(dim=1)).unsqueeze(-1) / 5.0
        fused = torch.cat(
            (
                z_attack,
                z_defense,
                z_attack - z_defense,
                z_attack * z_defense,
                z_cross,
                delta_power,
                delta_star,
            ),
            dim=-1,
        )
        hidden = self.trunk(fused)
        return {
            "win_prob": torch.sigmoid(self.win_head(hidden)).squeeze(-1),
            "uncertainty": F.softplus(self.uncertainty_head(hidden)).squeeze(-1) + 1e-4,
            "margin": self.margin_head(hidden).squeeze(-1),
            "duration": F.softplus(self.duration_head(hidden)).squeeze(-1) + 1e-4,
            "counter_residual": self.residual_head(hidden).squeeze(-1),
        }


class TorchSingleTeamScorer(SurrogateScorer):
    def __init__(self, model: SingleTeamWinrateModel, vocab: LoadoutVocab, *, device: torch.device | str | None = None) -> None:
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    def predict(self, attack: Team, defense: Team) -> SurrogatePrediction:
        self.model.eval()
        with torch.no_grad():
            batch = encode_team_batch((attack,), (defense,), self.vocab, device=self.device)
            output = self.model(batch)
        return SurrogatePrediction(
            win_prob=float(output["win_prob"][0].detach().cpu()),
            uncertainty=float(output["uncertainty"][0].detach().cpu()),
            margin=float(output["margin"][0].detach().cpu()),
            duration=float(output["duration"][0].detach().cpu()),
            counter_residual=float(output["counter_residual"][0].detach().cpu()),
        )


class SingleTeamEnsembleScorer(SurrogateScorer):
    def __init__(
        self,
        models: tuple[SingleTeamWinrateModel, ...],
        vocab: LoadoutVocab,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if not models:
            raise ValueError("models must be non-empty")
        self.models = tuple(models)
        self.vocab = vocab
        self.device = torch.device(device) if device is not None else next(models[0].parameters()).device

    def predict(self, attack: Team, defense: Team) -> SurrogatePrediction:
        outputs: list[dict[str, Tensor]] = []
        with torch.no_grad():
            batch = encode_team_batch((attack,), (defense,), self.vocab, device=self.device)
            for model in self.models:
                model.eval()
                outputs.append(model(batch))
        win_probs = [float(output["win_prob"][0].detach().cpu()) for output in outputs]
        aleatoric = [float(output["uncertainty"][0].detach().cpu()) for output in outputs]
        mean_prob = sum(win_probs) / len(win_probs)
        epistemic = math.sqrt(sum((value - mean_prob) ** 2 for value in win_probs) / len(win_probs))
        mean_aleatoric = sum(aleatoric) / len(aleatoric)
        return SurrogatePrediction(
            win_prob=mean_prob,
            uncertainty=math.sqrt(epistemic * epistemic + mean_aleatoric * mean_aleatoric),
            margin=sum(float(output["margin"][0].detach().cpu()) for output in outputs) / len(outputs),
            duration=sum(float(output["duration"][0].detach().cpu()) for output in outputs) / len(outputs),
            counter_residual=sum(float(output["counter_residual"][0].detach().cpu()) for output in outputs) / len(outputs),
        )


def save_single_team_model(path: Path, model: SingleTeamWinrateModel, vocab: LoadoutVocab) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vocab": vocab.to_dict(),
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_single_team_model(path: Path, *, device: torch.device | str | None = None) -> tuple[SingleTeamWinrateModel, LoadoutVocab]:
    payload = torch.load(Path(path), map_location=device or "cpu")
    vocab = LoadoutVocab.from_dict(payload["vocab"])
    config = SingleTeamWinrateModelConfig(**payload["config"])
    model = SingleTeamWinrateModel(vocab, config)
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, vocab


def binomial_nll(win_prob: Tensor, wins: Tensor, games: Tensor) -> Tensor:
    win_prob = win_prob.clamp(1e-6, 1.0 - 1e-6)
    return -(wins * torch.log(win_prob) + (games - wins) * torch.log1p(-win_prob)).mean()
