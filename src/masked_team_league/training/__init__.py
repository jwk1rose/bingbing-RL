"""训练与调度层，对应 tex §557-663、§1416-1439。

这里放单队模型、训练样本、checkpoint registry、模型选择和 recurring schedule。
生成 proposal 的网络与 teacher 逻辑属于 generation 层。
"""

from .checkpoints import CheckpointRegistry, ModelCheckpointRecord
from .model_selection import DatasetSplitManifest, SplitFileRecord, build_jsonl_split_manifest, select_best_checkpoint, write_split_manifest
from .schedule import (
    RecurringTrainingSchedulerState,
    ScheduledTrainingJob,
    TrainingSchedule,
    build_v4_training_schedule,
    load_training_schedule,
    run_training_schedule,
    write_training_schedule,
)
from .single_team_model import (
    LoadoutVocab,
    SingleTeamEnsembleScorer,
    SingleTeamWinrateModel,
    SingleTeamWinrateModelConfig,
    TorchSingleTeamScorer,
    load_single_team_model,
    save_single_team_model,
)
from .single_team_training import (
    HoldoutCalibrationReport,
    SingleTeamMatchupSample,
    TrainingHistory,
    build_holdout_calibration_report,
    evaluate_single_team_model,
    fit_single_team_calibrator,
    load_single_team_matchup_samples_jsonl,
    train_single_team_winrate_model,
)

__all__ = [
    "CheckpointRegistry",
    "DatasetSplitManifest",
    "HoldoutCalibrationReport",
    "LoadoutVocab",
    "ModelCheckpointRecord",
    "RecurringTrainingSchedulerState",
    "ScheduledTrainingJob",
    "SingleTeamEnsembleScorer",
    "SingleTeamMatchupSample",
    "SingleTeamWinrateModel",
    "SingleTeamWinrateModelConfig",
    "TorchSingleTeamScorer",
    "TrainingHistory",
    "TrainingSchedule",
    "build_holdout_calibration_report",
    "build_jsonl_split_manifest",
    "build_v4_training_schedule",
    "evaluate_single_team_model",
    "fit_single_team_calibrator",
    "load_single_team_matchup_samples_jsonl",
    "load_training_schedule",
    "load_single_team_model",
    "run_training_schedule",
    "save_single_team_model",
    "select_best_checkpoint",
    "train_single_team_winrate_model",
    "write_split_manifest",
    "write_training_schedule",
]
