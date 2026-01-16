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

        # Helper function to format percentage
        def fmt_pct(val):
            if isinstance(val, (int, float)):
                return f'{val:.1f}%'
            return str(val) if val else '-'

        # Helper function to format confusion matrix
        def fmt_confusion_matrix(metrics: dict, indent: str = "    "):
            cm_lines = []
            cm_lines.append(f"{indent}{'':20} {'actual_can':>12} {'actual_unc':>12} {'actual_cannot':>14}")
            cm_lines.append(f"{indent}predicted_can      {metrics.get('cm_can_can', 0):>12} {metrics.get('cm_can_uncertain', 0):>12} {metrics.get('cm_can_cannot', 0):>14}")
            cm_lines.append(f"{indent}predicted_uncertain{metrics.get('cm_uncertain_can', 0):>12} {metrics.get('cm_uncertain_uncertain', 0):>12} {metrics.get('cm_uncertain_cannot', 0):>14}")
            cm_lines.append(f"{indent}predicted_cannot   {metrics.get('cm_cannot_can', 0):>12} {metrics.get('cm_cannot_uncertain', 0):>12} {metrics.get('cm_cannot_cannot', 0):>14}")
            return cm_lines

        # ============================================================
        # STAGE 0: Pretrained Baseline
        # ============================================================
        if phase1.get('status') == 'completed':
            m1 = phase1.get('metrics', {})
            before_val = m1.get('before_val', {}) if isinstance(m1.get('before_val'), dict) else {}

            lines.append("")
            lines.append("=" * 80)
            lines.append("  [0] PRETRAINED BASELINE (预训练基线)")
            lines.append("=" * 80)

            baseline_qa = before_val.get('qa_accuracy', '-')
            baseline_jud = before_val.get('exact_match_rate', 0)

            lines.append(f"  QA Accuracy (Val):       {fmt_pct(baseline_qa)}")
            lines.append(f"  Judgment Accuracy (Val): {fmt_pct(baseline_jud)}")

            # Confusion matrix
            if before_val.get('cm_can_can') is not None:
                lines.append("")
                lines.append("  Confusion Matrix (Validation):")
                lines.extend(fmt_confusion_matrix(before_val))

        # ============================================================
        # STAGE 1: After Phase 1 (Judgment v1)
        # ============================================================
        if phase1.get('status') == 'completed':
            m1 = phase1.get('metrics', {})
            after_val = m1.get('after_val', {}) if isinstance(m1.get('after_val'), dict) else {}
            before_val = m1.get('before_val', {}) if isinstance(m1.get('before_val'), dict) else {}

            lines.append("")
            lines.append("=" * 80)
            lines.append("  [1] AFTER PHASE 1: Judgment v1 (判断能力 v1)")
            lines.append("=" * 80)

            p1_qa = after_val.get('qa_accuracy', '-')
            p1_jud = after_val.get('exact_match_rate', 0)
            baseline_jud = before_val.get('exact_match_rate', 0)
            jud_imp = p1_jud - baseline_jud if isinstance(p1_jud, (int, float)) and isinstance(baseline_jud, (int, float)) else 0

            lines.append(f"  QA Accuracy (Val):       {fmt_pct(p1_qa)}")
            lines.append(f"  Judgment Accuracy (Val): {fmt_pct(p1_jud)} ({jud_imp:+.1f}% from baseline)")

            # Confusion matrix
            if after_val.get('cm_can_can') is not None:
                lines.append("")
                lines.append("  Confusion Matrix (Validation):")
                lines.extend(fmt_confusion_matrix(after_val))

        # ============================================================
        # STAGE 2: After Phase 2 (Knowledge)
        # ============================================================
        if phase2.get('status') == 'completed':
            m2 = phase2.get('metrics', {})

            lines.append("")
            lines.append("=" * 80)
            lines.append("  [2] AFTER PHASE 2: Knowledge (知识学习)")
            lines.append("=" * 80)

            qa_before = m2.get('val_before_accuracy', 0) * 100
            qa_after = m2.get('val_after_accuracy', 0) * 100
            qa_imp = qa_after - qa_before

            lines.append(f"  QA Accuracy (Val):       {fmt_pct(qa_after)} ({qa_imp:+.1f}% from before)")
            lines.append(f"  Judgment Accuracy (Val): (not evaluated - knowledge only)")
            lines.append("")
            lines.append(f"  Training samples: {m2.get('qa_samples', 'N/A')}")

        # ============================================================
        # STAGE 3: After Phase 3 (Judgment v2)
        # ============================================================
        if phase3.get('status') == 'completed':
            m3 = phase3.get('metrics', {})

            lines.append("")
            lines.append("=" * 80)
            lines.append("  [3] AFTER PHASE 3: Judgment v2 (判断能力 v2 - Final)")
            lines.append("=" * 80)

            p3_qa = m3.get('qa_val_after', '-')
            p3_jud = m3.get('val_after_exact_match', 0) * 100

            # Get baseline for comparison
            baseline_jud = 0
            if phase1.get('status') == 'completed':
                m1 = phase1.get('metrics', {})
                before_val = m1.get('before_val', {}) if isinstance(m1.get('before_val'), dict) else {}
                baseline_jud = before_val.get('exact_match_rate', 0)

            total_jud_imp = p3_jud - baseline_jud

            lines.append(f"  QA Accuracy (Val):       {fmt_pct(p3_qa)}")
            lines.append(f"  Judgment Accuracy (Val): {fmt_pct(p3_jud)} ({total_jud_imp:+.1f}% from baseline)")

            # Ability distribution change
            orig_dist = m3.get('original_distribution', {})
            new_dist = m3.get('new_distribution', {})
            if orig_dist or new_dist:
                lines.append("")
                lines.append("  Ability Distribution Change (after knowledge learning):")
                lines.append(f"    {'Ability':<15} {'Before':>10} {'After':>10} {'Change':>10}")
                lines.append("    " + "-" * 50)
                for ability in ['can', 'uncertain', 'cannot']:
                    orig = orig_dist.get(ability, 0)
                    new = new_dist.get(ability, 0)
                    change = new - orig
                    lines.append(f"    {ability:<15} {orig:>10} {new:>10} {change:>+10}")

            # Confusion matrix (if available)
            # Note: Phase 3 metrics may store confusion matrix differently
            # We'll check for the standard keys
            val_after = m3.get('val_after', {}) if isinstance(m3.get('val_after'), dict) else {}
            if val_after.get('cm_can_can') is not None:
                lines.append("")
                lines.append("  Confusion Matrix (Validation):")
                lines.extend(fmt_confusion_matrix(val_after))

        # ============================================================
        # KEY IMPROVEMENTS SUMMARY
        # ============================================================
        lines.append("")
        lines.append("=" * 80)
        lines.append("  KEY IMPROVEMENTS (关键提升)")
        lines.append("=" * 80)

        if phase1.get('status') == 'completed':
            m1 = phase1.get('metrics', {})
            before_val = m1.get('before_val', {}) if isinstance(m1.get('before_val'), dict) else {}
            baseline_jud = before_val.get('exact_match_rate', 0)
            baseline_qa = before_val.get('qa_accuracy', '-')

            if phase3.get('status') == 'completed':
                m3 = phase3.get('metrics', {})
                final_jud = m3.get('val_after_exact_match', 0) * 100
                total_jud_imp = final_jud - baseline_jud
                lines.append(f"  Judgment: {baseline_jud:.1f}% → {final_jud:.1f}% ({total_jud_imp:+.1f}%)")
            else:
                after_val = m1.get('after_val', {}) if isinstance(m1.get('after_val'), dict) else {}
                p1_jud = after_val.get('exact_match_rate', 0)
                p1_imp = p1_jud - baseline_jud
                lines.append(f"  Judgment: {baseline_jud:.1f}% → {p1_jud:.1f}% ({p1_imp:+.1f}%)")

        if phase2.get('status') == 'completed':
            m2 = phase2.get('metrics', {})
            qa_before = m2.get('val_before_accuracy', 0) * 100
            qa_after = m2.get('val_after_accuracy', 0) * 100
            qa_imp = qa_after - qa_before
            lines.append(f"  QA Knowledge: {qa_before:.1f}% → {qa_after:.1f}% ({qa_imp:+.1f}%)")

        if phase3.get('status') == 'completed':
            m3 = phase3.get('metrics', {})
            orig_dist = m3.get('original_distribution', {})
            new_dist = m3.get('new_distribution', {})
            if orig_dist and new_dist:
                orig_can = orig_dist.get('can', 0)
                new_can = new_dist.get('can', 0)
                lines.append(f"  Confidence ('can' count): {orig_can} → {new_can} ({new_can - orig_can:+d})")

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
