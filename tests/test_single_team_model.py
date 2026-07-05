from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from masked_team_league.domain import Team
from masked_team_league.training.single_team_model import (
    LoadoutVocab,
    SingleTeamEnsembleScorer,
    SingleTeamWinrateModel,
    SingleTeamWinrateModelConfig,
    TorchSingleTeamScorer,
    encode_team_batch,
    load_single_team_model,
    save_single_team_model,
)


def test_single_team_model_outputs_documented_heads(loadouts):
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    model = SingleTeamWinrateModel(vocab, config)
    batch = encode_team_batch((Team(loadouts[:5]),), (Team(loadouts[5:10]),), vocab)

    with torch.no_grad():
        output = model(batch)

    assert set(output) == {"win_prob", "uncertainty", "margin", "duration", "counter_residual"}
    assert output["win_prob"].shape == (1,)
    assert 0.0 < float(output["win_prob"][0]) < 1.0
    assert float(output["uncertainty"][0]) > 0.0
    assert float(output["duration"][0]) > 0.0


def test_single_team_model_is_position_aware(loadouts):
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    torch.manual_seed(7)
    model = SingleTeamWinrateModel(vocab, config)
    attack = Team(loadouts[:5])
    defense = Team(loadouts[5:10])
    reordered_attack = Team((loadouts[1], loadouts[0], loadouts[2], loadouts[3], loadouts[4]))

    with torch.no_grad():
        base = model(encode_team_batch((attack,), (defense,), vocab))["win_prob"]
        shifted = model(encode_team_batch((reordered_attack,), (defense,), vocab))["win_prob"]

    assert not torch.allclose(base, shifted)


def test_single_team_model_is_unique_equip_star_aware(loadouts):
    changed = replace(loadouts[0], unique_equip_star=5 if loadouts[0].unique_equip_star != 5 else 3)
    pool = (changed, *loadouts[1:])
    vocab = LoadoutVocab.from_loadouts(pool)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    torch.manual_seed(11)
    model = SingleTeamWinrateModel(vocab, config)
    defense = Team(loadouts[5:10])

    with torch.no_grad():
        base = model(encode_team_batch((Team(loadouts[:5]),), (defense,), vocab))["win_prob"]
        starred = model(encode_team_batch((Team((changed, *loadouts[1:5])),), (defense,), vocab))["win_prob"]

    assert not torch.allclose(base, starred)


def test_torch_single_team_scorer_returns_surrogate_prediction(loadouts):
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    model = SingleTeamWinrateModel(vocab, config)
    scorer = TorchSingleTeamScorer(model, vocab)

    prediction = scorer.predict(Team(loadouts[:5]), Team(loadouts[5:10]))

    assert 0.0 < prediction.win_prob < 1.0
    assert prediction.uncertainty > 0.0
    assert prediction.duration > 0.0


def test_single_team_ensemble_scorer_combines_epistemic_uncertainty(loadouts):
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    models = (SingleTeamWinrateModel(vocab, config), SingleTeamWinrateModel(vocab, config))
    scorer = SingleTeamEnsembleScorer(models, vocab)

    prediction = scorer.predict(Team(loadouts[:5]), Team(loadouts[5:10]))

    assert 0.0 < prediction.win_prob < 1.0
    assert prediction.uncertainty > 0.0


def test_single_team_model_save_and_load_round_trips(tmp_path, loadouts):
    vocab = LoadoutVocab.from_loadouts(loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=32, hero_dim=16, equip_dim=8, star_dim=4, bucket_dim=4, heads=4, layers=1)
    model = SingleTeamWinrateModel(vocab, config)
    path = tmp_path / "single_team.pt"

    save_single_team_model(path, model, vocab)
    loaded_model, loaded_vocab = load_single_team_model(path)

    assert loaded_vocab.hero_to_index == vocab.hero_to_index
    prediction = TorchSingleTeamScorer(loaded_model, loaded_vocab).predict(Team(loadouts[:5]), Team(loadouts[5:10]))
    assert 0.0 < prediction.win_prob < 1.0
