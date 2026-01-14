"""
Multi-phase Pipeline Manager - Orchestrate the full training workflow.

Supports:
- Saving/loading state for each phase
- Resuming from any phase
- Tracking metrics across phases
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class PhaseResult:
    """Result of a single phase."""
    phase: str
    status: str  # "completed", "failed", "in_progress"
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PipelineState:
    """Full pipeline state for checkpointing."""
    experiment_name: str
    base_model: str
    current_phase: int = 0
    phases: Dict[str, PhaseResult] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        """Save state to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert PhaseResult objects to dicts
        state_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PipelineState':
        """Load state from JSON."""
        with open(path) as f:
            data = json.load(f)

        # Convert dicts back to PhaseResult objects
        if 'phases' in data:
            phases = {k: PhaseResult(**v) for k, v in data['phases'].items()}
            data['phases'] = phases

        return cls(**data)


class MultiPhasePipeline:
    """
    Manage the three-phase training pipeline.

    Phase 1: Initial judgment training (existing steps 1-4)
    Phase 2: Knowledge learning (new)
    Phase 3: Updated judgment training (new)
    """

    def __init__(
        self,
        experiment_name: str,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./experiments",
        config: Dict = None
    ):
        self.experiment_name = experiment_name
        self.base_model = base_model
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.output_dir / "pipeline_state.json"

        # Load existing state or create new
        if self.state_path.exists():
            self.state = PipelineState.load(str(self.state_path))
            print(f"Loaded existing pipeline state from {self.state_path}")
            print(f"Current phase: {self.state.current_phase}")
        else:
            self.state = PipelineState(
                experiment_name=experiment_name,
                base_model=base_model,
                current_phase=0,
                phases={},
                config=config or {}
            )
            self._save_state()

    def _save_state(self):
        """Save current state."""
        self.state.save(str(self.state_path))

    def get_phase_output_dir(self, phase_name: str) -> Path:
        """Get output directory for a phase."""
        phase_dir = self.output_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def record_phase_result(
        self,
        phase_name: str,
        status: str,
        metrics: Dict = None,
        output_paths: Dict = None
    ):
        """Record result of a phase."""
        result = PhaseResult(
            phase=phase_name,
            status=status,
            metrics=metrics or {},
            output_paths=output_paths or {},
        )
        self.state.phases[phase_name] = result
        self._save_state()

    def get_phase_result(self, phase_name: str) -> Optional[PhaseResult]:
        """Get result of a specific phase."""
        return self.state.phases.get(phase_name)

    def get_summary(self) -> Dict:
        """Get summary of all phases."""
        return {
            "experiment": self.experiment_name,
            "base_model": self.base_model,
            "current_phase": self.state.current_phase,
            "output_dir": str(self.output_dir),
            "phases": {
                name: {
                    "status": result.status,
                    "metrics": result.metrics,
                    "timestamp": result.timestamp
                }
                for name, result in self.state.phases.items()
            }
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print(f"Experiment: {summary['experiment']}")
        print(f"Base Model: {summary['base_model']}")
        print(f"Output Dir: {summary['output_dir']}")
        print("=" * 60)

        for phase_name, phase_info in summary['phases'].items():
            print(f"\n{phase_name}:")
            print(f"  Status: {phase_info['status']}")
            if phase_info['metrics']:
                for k, v in phase_info['metrics'].items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        print("=" * 60)


def create_experiment(
    name: str,
    base_model: str,
    output_dir: str = "./experiments",
    **config
) -> MultiPhasePipeline:
    """Create a new experiment pipeline."""
    return MultiPhasePipeline(
        experiment_name=name,
        base_model=base_model,
        output_dir=output_dir,
        config=config
    )


def load_experiment(
    name: str,
    output_dir: str = "./experiments"
) -> MultiPhasePipeline:
    """Load an existing experiment."""
    state_path = Path(output_dir) / name / "pipeline_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Experiment '{name}' not found at {state_path}")

    state = PipelineState.load(str(state_path))
    return MultiPhasePipeline(
        experiment_name=state.experiment_name,
        base_model=state.base_model,
        output_dir=output_dir,
        config=state.config
    )
