from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from masked_team_league.domain import MatchFormat
from masked_team_league.generation.proposal_networks import (
    AttackGenerationNetwork,
    DefenseRosterGenerationNetwork,
    GenerationContextEncoder,
    MaskSelectionNetwork,
    ProposalNetworkConfig,
    apply_legal_action_mask,
    beam_search_proposal_tokens,
    build_causal_attention_mask,
    proposal_distillation_loss,
    sample_proposal_tokens,
)


def test_causal_mask_blocks_future_tokens():
    mask = build_causal_attention_mask(4)

    assert mask.shape == (4, 4)
    assert mask[3, 0] == 0.0
    assert mask[0, 0] == 0.0
    assert math.isinf(float(mask[0, 1])) and float(mask[0, 1]) < 0.0


def test_apply_legal_action_mask_sets_illegal_logits_to_negative_infinity():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    legal = torch.tensor([[True, False, True]])

    masked = apply_legal_action_mask(logits, legal)

    assert masked[0, 0] == 1.0
    assert math.isinf(float(masked[0, 1])) and float(masked[0, 1]) < 0.0
    assert masked[0, 2] == 3.0


def test_attack_and_defense_generation_networks_apply_action_mask():
    config = ProposalNetworkConfig(loadout_count=7, model_dim=32, heads=4, layers=1, max_slots=15)
    selected = torch.tensor([[0, 2, 3]])
    legal = torch.tensor([[True, False, True, True, False, True, True]])
    attack_net = AttackGenerationNetwork(config)
    defense_net = DefenseRosterGenerationNetwork(config)

    with torch.no_grad():
        attack_out = attack_net(selected, legal)
        defense_out = defense_net(selected, legal)

    assert attack_out.masked_logits.shape == (1, 7)
    assert defense_out.masked_logits.shape == (1, 7)
    assert math.isinf(float(attack_out.masked_logits[0, 1]))
    assert math.isinf(float(defense_out.masked_logits[0, 4]))
    assert attack_out.value_estimates.shape == (1,)
    assert defense_out.value_estimates.shape == (1,)
    assert attack_out.anti_meta_residual_estimates.shape == (1,)
    assert defense_out.anti_meta_residual_estimates.shape == (1,)


def test_generation_context_encoder_pools_observation_belief_and_pool_features():
    config = ProposalNetworkConfig(loadout_count=9, model_dim=32, heads=4, layers=1, max_slots=15)
    encoder = GenerationContextEncoder(config, numeric_feature_dim=3)
    observation_tokens = torch.tensor([[0, 1, 2, 0, 3]])
    observation_hidden = torch.tensor([[True, False, False, True, False]])
    belief_tokens = torch.tensor([[[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]])
    belief_weights = torch.tensor([[0.75, 0.25]])
    pool_tokens = torch.tensor([[1, 2, 3, 4]])
    numeric_features = torch.tensor([[0.1, 0.2, 0.3]])

    output = encoder(
        observation_tokens,
        observation_hidden_mask=observation_hidden,
        belief_token_ids=belief_tokens,
        belief_weights=belief_weights,
        pool_token_ids=pool_tokens,
        numeric_features=numeric_features,
    )

    assert output.context_vector.shape == (1, 32)
    assert output.context_memory.shape[0] == 1
    assert output.context_memory.shape[2] == 32
    assert torch.isfinite(output.context_vector).all()


def test_generation_network_forward_accepts_context_vector():
    config = ProposalNetworkConfig(loadout_count=7, model_dim=32, heads=4, layers=1, max_slots=15)
    selected = torch.tensor([[0, 2, 3]])
    legal = torch.tensor([[True, True, True, True, True, True, True]])
    network = AttackGenerationNetwork(config)

    with torch.no_grad():
        no_context = network(selected, legal, context_vector=torch.zeros((1, 32)))
        with_context = network(selected, legal, context_vector=torch.ones((1, 32)))

    assert no_context.masked_logits.shape == with_context.masked_logits.shape
    assert not torch.allclose(no_context.masked_logits, with_context.masked_logits)


def test_beam_search_proposal_tokens_respects_per_step_legal_masks():
    config = ProposalNetworkConfig(loadout_count=6, model_dim=32, heads=4, layers=1, max_slots=8)
    network = AttackGenerationNetwork(config)
    forced = (2, 0, 4)

    def legal_mask(prefix):
        mask = [False] * config.loadout_count
        mask[forced[len(prefix)]] = True
        return tuple(mask)

    sequences = beam_search_proposal_tokens(network, legal_action_mask_fn=legal_mask, max_steps=3, beam_size=3)

    assert sequences[0].token_ids == forced
    assert sequences[0].log_prob <= 0.0
    assert len(sequences) == 1


def test_sample_proposal_tokens_respects_per_step_legal_masks():
    config = ProposalNetworkConfig(loadout_count=6, model_dim=32, heads=4, layers=1, max_slots=8)
    network = AttackGenerationNetwork(config)
    forced = (1, 3, 5)

    def legal_mask(prefix):
        mask = [False] * config.loadout_count
        mask[forced[len(prefix)]] = True
        return tuple(mask)

    sequence = sample_proposal_tokens(network, legal_action_mask_fn=legal_mask, max_steps=3, seed=123)

    assert sequence.token_ids == forced
    assert sequence.log_prob <= 0.0


def test_proposal_distillation_loss_combines_bc_value_and_gap_terms():
    logits = torch.tensor([[[3.0, 0.0, -1.0], [0.0, 4.0, -1.0]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False], [True, True, False]]])
    targets = torch.tensor([[0, 1]])
    value_estimates = torch.tensor([0.2], requires_grad=True)
    value_targets = torch.tensor([0.8])
    gap_estimates = torch.tensor([-0.1], requires_grad=True)
    gap_targets = torch.tensor([0.3])

    losses = proposal_distillation_loss(
        logits,
        targets,
        legal_action_mask=legal,
        value_estimates=value_estimates,
        value_targets=value_targets,
        gap_estimates=gap_estimates,
        gap_targets=gap_targets,
        value_weight=2.0,
        gap_weight=0.5,
    )

    assert losses.behavior_cloning_loss.item() < 0.1
    assert losses.value_loss.item() > 0.0
    assert losses.gap_loss.item() > 0.0
    assert torch.isclose(
        losses.total_loss,
        losses.behavior_cloning_loss + 2.0 * losses.value_loss + 0.5 * losses.gap_loss,
    )
    losses.total_loss.backward()
    assert logits.grad is not None
    assert value_estimates.grad is not None
    assert gap_estimates.grad is not None


def test_proposal_distillation_loss_can_supervise_anti_meta_residual_head():
    logits = torch.tensor([[[3.0, 0.0, -1.0]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False]]])
    targets = torch.tensor([[0]])
    residual_estimates = torch.tensor([0.1], requires_grad=True)
    residual_targets = torch.tensor([0.6])

    losses = proposal_distillation_loss(
        logits,
        targets,
        legal_action_mask=legal,
        anti_meta_residual_estimates=residual_estimates,
        anti_meta_residual_targets=residual_targets,
        anti_meta_residual_weight=3.0,
    )

    assert losses.anti_meta_residual_loss.item() > 0.0
    assert torch.isclose(
        losses.total_loss,
        losses.behavior_cloning_loss + 3.0 * losses.anti_meta_residual_loss,
    )
    losses.total_loss.backward()
    assert residual_estimates.grad is not None


def test_proposal_distillation_loss_rejects_illegal_teacher_action():
    logits = torch.zeros((1, 1, 3))
    legal = torch.tensor([[[True, False, True]]])
    targets = torch.tensor([[1]])

    with pytest.raises(ValueError, match="teacher target is illegal"):
        proposal_distillation_loss(logits, targets, legal_action_mask=legal)


def test_mask_selection_network_constrained_selector_respects_limits_and_leakage():
    fmt = MatchFormat(3)
    network = MaskSelectionNetwork(feature_dim=3, hidden_dim=8)
    slot_scores = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [0.9, 0.8, 0.7, 0.6, 0.5],
        ]
    )
    position_leakage = torch.zeros_like(slot_scores)
    position_leakage[0, 0] = 100.0

    mask = network.select_mask(slot_scores, fmt, position_leakage=position_leakage)

    assert len(mask) == 3
    assert all(sum(row) <= fmt.max_hidden_per_team for row in mask)
    assert sum(sum(row) for row in mask) <= fmt.max_hidden_total
    assert mask[0][0] == 0
    assert sum(sum(row) for row in mask) == 6
