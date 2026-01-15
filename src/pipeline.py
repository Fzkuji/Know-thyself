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

    def print_summary(self, save_to_file: bool = True):
        """Print formatted summary and optionally save to file."""
        lines = self._generate_summary_lines()

        # Print to console
        for line in lines:
            print(line)

        # Save to file
        if save_to_file:
            self.save_summary(lines)

    def save_summary(self, lines: list = None):
        """Save summary to a text file in the experiment directory."""
        if lines is None:
            lines = self._generate_summary_lines()

        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nSummary saved to: {summary_path}")

    def _generate_summary_lines(self) -> list:
        """Generate summary as a list of lines."""
        lines = []
        summary = self.get_summary()

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  EXPERIMENT SUMMARY: {summary['experiment']}")
        lines.append("=" * 80)
        lines.append(f"  Base Model: {summary['base_model']}")
        lines.append(f"  Output Dir: {summary['output_dir']}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Check which phases are completed
        phases = summary['phases']
        phase1 = phases.get('phase1_judgment', {})
        phase2 = phases.get('phase2_knowledge', {})
        phase3 = phases.get('phase3_judgment', {})

        # Generate comprehensive cross-phase comparison
        lines.extend(self._generate_comprehensive_summary_lines(phase1, phase2, phase3))

        return lines

    def _generate_comprehensive_summary_lines(self, phase1: dict, phase2: dict, phase3: dict) -> list:
        """Generate comprehensive cross-phase comparison as lines."""
        lines = []

        # ===== TABLE 1: JUDGMENT ACCURACY =====
        lines.append("")
        lines.append("=" * 80)
        lines.append("  TABLE 1: JUDGMENT ACCURACY (across all stages)")
        lines.append("=" * 80)
        lines.append(f"  {'Stage':<40} {'TRAIN':>15} {'VALIDATION':>15}")
        lines.append("  " + "-" * 76)

        # Collect judgment accuracy data
        judgment_data = []

        if phase1.get('status') == 'completed':
            m1 = phase1.get('metrics', {})
            before_train = m1.get('before_train', {})
            before_val = m1.get('before_val', {})
            after_train = m1.get('after_train', {})
            after_val = m1.get('after_val', {})

            bt = before_train.get('exact_match_rate', 0) if isinstance(before_train, dict) else 0
            bv = before_val.get('exact_match_rate', 0) if isinstance(before_val, dict) else 0
            at = after_train.get('exact_match_rate', 0) if isinstance(after_train, dict) else 0
            av = after_val.get('exact_match_rate', 0) if isinstance(after_val, dict) else 0

            judgment_data.append(('Baseline (pretrained)', bt, bv))
            judgment_data.append(('After Phase 1 (Judgment v1)', at, av))

        if phase2.get('status') == 'completed':
            judgment_data.append(('After Phase 2 (Knowledge)', '-', '-'))

        if phase3.get('status') == 'completed':
            m3 = phase3.get('metrics', {})
            bt = m3.get('train_before_exact_match', 0) * 100
            bv = m3.get('val_before_exact_match', 0) * 100
            at = m3.get('train_after_exact_match', 0) * 100
            av = m3.get('val_after_exact_match', 0) * 100

            judgment_data.append(('Before Phase 3 (Knowledge only)', bt, bv))
            judgment_data.append(('After Phase 3 (Judgment v2)', at, av))

        for stage, train_acc, val_acc in judgment_data:
            train_str = f'{train_acc:.1f}%' if isinstance(train_acc, (int, float)) else train_acc
            val_str = f'{val_acc:.1f}%' if isinstance(val_acc, (int, float)) else val_acc
            lines.append(f"  {stage:<40} {train_str:>15} {val_str:>15}")

        # ===== TABLE 2: QA ACCURACY =====
        lines.append("")
        lines.append("=" * 80)
        lines.append("  TABLE 2: QA ACCURACY (across all stages)")
        lines.append("=" * 80)
        lines.append(f"  {'Stage':<40} {'TRAIN':>15} {'VALIDATION':>15}")
        lines.append("  " + "-" * 76)

        # Collect QA accuracy data
        qa_data = []

        if phase1.get('status') == 'completed':
            m1 = phase1.get('metrics', {})
            before_train = m1.get('before_train', {})
            before_val = m1.get('before_val', {})
            after_train = m1.get('after_train', {})
            after_val = m1.get('after_val', {})

            bt_qa = before_train.get('qa_accuracy', '-') if isinstance(before_train, dict) else '-'
            bv_qa = before_val.get('qa_accuracy', '-') if isinstance(before_val, dict) else '-'
            at_qa = after_train.get('qa_accuracy', '-') if isinstance(after_train, dict) else '-'
            av_qa = after_val.get('qa_accuracy', '-') if isinstance(after_val, dict) else '-'

            qa_data.append(('Baseline (pretrained)', bt_qa, bv_qa))
            qa_data.append(('After Phase 1 (Judgment v1)', at_qa, av_qa))

        if phase2.get('status') == 'completed':
            m2 = phase2.get('metrics', {})
            bt = m2.get('train_before_accuracy', 0) * 100
            bv = m2.get('val_before_accuracy', 0) * 100
            at = m2.get('train_after_accuracy', 0) * 100
            av = m2.get('val_after_accuracy', 0) * 100

            qa_data.append(('Before Phase 2 (pre-Knowledge)', bt, bv))
            qa_data.append(('After Phase 2 (Knowledge)', at, av))

        if phase3.get('status') == 'completed':
            m3 = phase3.get('metrics', {})
            bt = m3.get('qa_train_before', '-')
            bv = m3.get('qa_val_before', '-')
            at = m3.get('qa_train_after', '-')
            av = m3.get('qa_val_after', '-')

            qa_data.append(('Before Phase 3 (Knowledge only)', bt, bv))
            qa_data.append(('After Phase 3 (Judgment v2)', at, av))

        for stage, train_acc, val_acc in qa_data:
            train_str = f'{train_acc:.1f}%' if isinstance(train_acc, (int, float)) else str(train_acc)
            val_str = f'{val_acc:.1f}%' if isinstance(val_acc, (int, float)) else str(val_acc)
            lines.append(f"  {stage:<40} {train_str:>15} {val_str:>15}")

        # ===== ABILITY DISTRIBUTION (if Phase 3 completed) =====
        if phase3.get('status') == 'completed':
            m3 = phase3.get('metrics', {})
            orig_dist = m3.get('original_distribution', {})
            new_dist = m3.get('new_distribution', {})
            if orig_dist and new_dist:
                lines.append("")
                lines.append("=" * 80)
                lines.append("  ABILITY DISTRIBUTION CHANGE (after knowledge learning)")
                lines.append("=" * 80)
                lines.append(f"  {'Ability':<40} {'Before':>15} {'After':>15}")
                lines.append("  " + "-" * 76)
                for ability in ['can', 'uncertain', 'cannot']:
                    orig = orig_dist.get(ability, 0)
                    new = new_dist.get(ability, 0)
                    lines.append(f"  {ability:<40} {orig:>15} {new:>15}")

        # ===== KEY INSIGHTS =====
        lines.append("")
        lines.append("=" * 80)
        lines.append("  KEY INSIGHTS")
        lines.append("=" * 80)

        # Judgment improvement
        if phase1.get('status') == 'completed':
            m1 = phase1.get('metrics', {})
            before_val = m1.get('before_val', {})
            baseline_judgment = before_val.get('exact_match_rate', 0) if isinstance(before_val, dict) else 0

            if phase3.get('status') == 'completed':
                m3 = phase3.get('metrics', {})
                final_judgment = m3.get('val_after_exact_match', 0) * 100
                total_improvement = final_judgment - baseline_judgment
                lines.append("")
                lines.append("  Judgment Accuracy (Validation):")
                lines.append(f"    Baseline → Final: {baseline_judgment:.1f}% → {final_judgment:.1f}% ({total_improvement:+.1f}%)")
            else:
                after_val = m1.get('after_val', {})
                p1_judgment = after_val.get('exact_match_rate', 0) if isinstance(after_val, dict) else 0
                p1_improvement = p1_judgment - baseline_judgment
                lines.append("")
                lines.append("  Judgment Accuracy (Validation after Phase 1):")
                lines.append(f"    Baseline → After P1: {baseline_judgment:.1f}% → {p1_judgment:.1f}% ({p1_improvement:+.1f}%)")

        # QA improvement
        if phase2.get('status') == 'completed':
            m2 = phase2.get('metrics', {})
            qa_before = m2.get('val_before_accuracy', 0) * 100
            qa_after = m2.get('val_after_accuracy', 0) * 100
            qa_improvement = qa_after - qa_before
            lines.append("")
            lines.append("  QA Accuracy (Validation after Phase 2):")
            lines.append(f"    Before → After: {qa_before:.1f}% → {qa_after:.1f}% ({qa_improvement:+.1f}%)")

        # Model confidence
        if phase3.get('status') == 'completed':
            m3 = phase3.get('metrics', {})
            orig_dist = m3.get('original_distribution', {})
            new_dist = m3.get('new_distribution', {})
            if orig_dist and new_dist:
                orig_can = orig_dist.get('can', 0)
                new_can = new_dist.get('can', 0)
                if new_can > orig_can:
                    lines.append("")
                    lines.append("  Model Confidence Change:")
                    lines.append(f"    'can' answers: {orig_can} → {new_can} (+{new_can - orig_can})")

        lines.append("")
        lines.append("=" * 80)

        return lines


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
