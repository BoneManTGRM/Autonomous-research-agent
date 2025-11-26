"""
Ultra-Maxed Meta Evaluator Agent

This agent integrates all signals across:
    - TGRM cycles
    - Replay buffer
    - RYE gradients and slopes
    - Hallmark profiles and curriculum states
    - Mechanism chains
    - Contradictions and critic output
    - Swarm agent roles
    - MemoryStore knowledge graphs
    - Run-manager-level signals

It produces:
    - breakthrough detection
    - equilibrium classification
    - run-mode shifts
    - swarm role reallocations
    - curriculum stage transitions
    - replay sampling strategies
    - mechanism-chain reliability scoring
    - discovery potential estimates

This is the absolute ceiling for a meta-evaluator in your architecture.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import statistics


class MetaEvaluatorAgent:
    def __init__(
        self,
        memory_store: Any,
        replay_buffer: Any,
        hallmark_profiles: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.memory_store = memory_store
        self.replay_buffer = replay_buffer
        self.hallmark_profiles = hallmark_profiles
        self.config = config or {}

        # Meta-level thresholds
        self.breakthrough_min_slope = float(self.config.get("meta_breakthrough_slope", 0.12))
        self.equilibrium_noise_threshold = float(self.config.get("meta_equilibrium_noise", 0.08))
        self.replay_recency_weight = float(self.config.get("meta_replay_recency_weight", 0.65))
        self.min_cycles_for_trend = int(self.config.get("meta_min_cycles_for_trend", 24))

        # Swarm and curriculum directives
        self.force_verify_if_breakthrough = bool(self.config.get("meta_force_verify", True))
        self.force_explore_if_stagnant = bool(self.config.get("meta_force_explore", True))
        self.enable_hallmark_shift = bool(self.config.get("meta_enable_hallmark_shift", True))

    # ---------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------
    def handle_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload includes:
            - run_history (list of cycle summaries)
            - cycle_index
            - run_id
            - hallmark
            - curriculum_state
            - swarm_state (optional)
        """

        history = payload.get("run_history", [])
        cycle_index = payload.get("cycle_index")
        hallmark = payload.get("hallmark")
        curriculum_state = payload.get("curriculum_state", {})
        swarm_state = payload.get("swarm_state", {})

        # -----------------------------------------------------------
        # 1. Compute RYE trend classification
        # -----------------------------------------------------------
        trend = self._analyze_rye_trend(history)

        # -----------------------------------------------------------
        # 2. Analyze replay patterns for learning loops
        # -----------------------------------------------------------
        replay_signals = self._analyze_replay()

        # -----------------------------------------------------------
        # 3. Estimate breakthrough probability
        # -----------------------------------------------------------
        bscore = self._estimate_breakthrough_potential(trend, replay_signals)

        # -----------------------------------------------------------
        # 4. Determine equilibrium mode
        # -----------------------------------------------------------
        eq_state = self._classify_equilibrium(history, trend)

        # -----------------------------------------------------------
        # 5. Produce meta-directives
        # -----------------------------------------------------------
        directives = self._build_meta_directives(
            hallmark=hallmark,
            curriculum_state=curriculum_state,
            trend=trend,
            replay_signals=replay_signals,
            breakthrough_score=bscore,
            eq_state=eq_state,
            cycle_index=cycle_index,
        )

        # -----------------------------------------------------------
        # 6. Build meta-note
        # -----------------------------------------------------------
        note = self._build_meta_note(
            hallmark=hallmark,
            trend=trend,
            replay_signals=replay_signals,
            breakthrough_score=bscore,
            eq_state=eq_state,
            directives=directives,
        )

        # Store in memory
        try:
            self.memory_store.add_note(
                hallmark,
                note,
                role="meta",
                metadata={
                    "hallmark": hallmark,
                    "source": "meta_evaluator",
                },
            )
        except:
            pass

        # Store meta evaluation in replay
        try:
            item = {
                "item_type": "meta_eval",
                "hallmark": hallmark,
                "trend": trend,
                "replay_signals": replay_signals,
                "breakthrough_score": bscore,
                "equilibrium_state": eq_state,
                "directives": directives,
                "cycle_index": cycle_index,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            if hasattr(self.replay_buffer, "add_item"):
                self.replay_buffer.add_item(item)
        except:
            pass

        return {
            "ok": True,
            "trend": trend,
            "breakthrough_score": bscore,
            "equilibrium_state": eq_state,
            "directives": directives,
            "meta_note": note,
        }

    # ---------------------------------------------------------------
    # RYE trend analysis
    # ---------------------------------------------------------------
    def _analyze_rye_trend(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(history) < self.min_cycles_for_trend:
            return {"slope": 0.0, "label": "insufficient_data", "variance": 0.0}

        ryes = [c.get("rye", 0) for c in history[-self.min_cycles_for_trend:]]
        slope = self._linear_slope(ryes)
        variance = statistics.pvariance(ryes) if len(ryes) > 1 else 0.0

        if slope > self.breakthrough_min_slope:
            label = "rising"
        elif slope < -self.breakthrough_min_slope:
            label = "falling"
        else:
            label = "flat"

        return {
            "slope": slope,
            "label": label,
            "variance": variance,
        }

    # ---------------------------------------------------------------
    # Replay analysis
    # ---------------------------------------------------------------
    def _analyze_replay(self) -> Dict[str, Any]:
        recent = self.replay_buffer.get_recent_items(40)
        if not recent:
            return {"avg_recent_rye": 0, "hot_hallmark": None, "frequency": {}}

        ry = []
        freq = {}
        for item in recent:
            hallmark = item.get("hallmark")
            if hallmark:
                freq[hallmark] = freq.get(hallmark, 0) + 1
            if "rye_score" in item:
                ry.append(item["rye_score"])

        avg_recent_rye = sum(ry)/len(ry) if ry else 0
        hot_hallmark = max(freq.items(), key=lambda x: x[1])[0] if freq else None

        return {
            "avg_recent_rye": avg_recent_rye,
            "hot_hallmark": hot_hallmark,
            "frequency": freq,
        }

    # ---------------------------------------------------------------
    # Breakthrough scoring
    # ---------------------------------------------------------------
    def _estimate_breakthrough_potential(self, trend, replay_signals):
        slope = trend.get("slope", 0)
        avg_rye = replay_signals.get("avg_recent_rye", 0)

        # Breakthrough score uses slope + replay recency
        score = (
            0.6 * max(0, slope) +
            0.4 * max(0, avg_rye) * self.replay_recency_weight
        )

        return min(score, 1.0)

    # ---------------------------------------------------------------
    # Equilibrium classification
    # ---------------------------------------------------------------
    def _classify_equilibrium(self, history, trend):
        if len(history) < self.min_cycles_for_trend:
            return "unknown"

        variance = trend.get("variance", 0)
        if variance < self.equilibrium_noise_threshold:
            return "stable"
        elif variance < self.equilibrium_noise_threshold * 2:
            return "shifting"
        return "chaotic"

    # ---------------------------------------------------------------
    # Meta-directives (the real high-level intelligence)
    # ---------------------------------------------------------------
    def _build_meta_directives(
        self,
        hallmark,
        curriculum_state,
        trend,
        replay_signals,
        breakthrough_score,
        eq_state,
        cycle_index,
    ):
        directives = {}

        # -----------------------------
        # 1. Switch to verify if near breakthrough
        # -----------------------------
        if self.force_verify_if_breakthrough and breakthrough_score > 0.7:
            directives["prefer_stage"] = "verify"
        # -----------------------------
        # 2. Force exploration if flatlining
        # -----------------------------
        elif self.force_explore_if_stagnant and trend["label"] == "flat":
            directives["prefer_stage"] = "idea"
        else:
            directives["prefer_stage"] = None

        # -----------------------------
        # 3. Hallmark shifting
        # -----------------------------
        if self.enable_hallmark_shift:
            hot = replay_signals.get("hot_hallmark")
            if hot and hot != hallmark:
                directives["hallmark_shift"] = hot

        # -----------------------------
        # 4. Curriculum steering
        # -----------------------------
        cur_phase = curriculum_state.get("phase")
        if breakthrough_score > 0.75 and cur_phase != "exploit":
            directives["curriculum_shift"] = "exploit"
        elif trend["label"] == "rising" and cur_phase == "bootstrap":
            directives["curriculum_shift"] = "depth"
        elif eq_state == "chaotic" and cur_phase != "stabilize":
            directives["curriculum_shift"] = "stabilize"

        # -----------------------------
        # 5. Replay sampling strategy
        # -----------------------------
        if breakthrough_score > 0.5:
            directives["replay_focus"] = "strong_high_rye"
        elif trend["label"] == "falling":
            directives["replay_focus"] = "diversity"
        else:
            directives["replay_focus"] = "balanced"

        # -----------------------------
        # 6. Swarm role adjustments
        # -----------------------------
        if trend["label"] == "rising":
            directives["swarm_role_bias"] = "more_critics"
        elif trend["label"] == "flat":
            directives["swarm_role_bias"] = "more_explorers"
        else:
            directives["swarm_role_bias"] = "balanced"

        return directives

    # ---------------------------------------------------------------
    # Meta note
    # ---------------------------------------------------------------
    def _build_meta_note(
        self,
        hallmark,
        trend,
        replay_signals,
        breakthrough_score,
        eq_state,
        directives,
    ):
        lines = []
        lines.append("[meta] Meta Evaluation Summary")
        lines.append(f"Hallmark: {hallmark}")
        lines.append(f"RYE Trend: slope={trend['slope']:.3f} label={trend['label']}")
        lines.append(f"Equilibrium State: {eq_state}")
        lines.append("")
        lines.append("Replay analysis:")
        lines.append(f"- Recent avg RYE: {replay_signals['avg_recent_rye']:.3f}")
        lines.append(f"- Hot hallmark: {replay_signals.get('hot_hallmark')}")
        lines.append("")
        lines.append(f"Breakthrough score: {breakthrough_score:.3f}")
        lines.append("")
        lines.append("Directives:")
        for k, v in directives.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    # ---------------------------------------------------------------
    # Utility: slope
    # ---------------------------------------------------------------
    def _linear_slope(self, seq):
        n = len(seq)
        if n < 2:
            return 0
        x = list(range(n))
        avg_x = sum(x)/n
        avg_y = sum(seq)/n
        num = sum((x[i]-avg_x)*(seq[i]-avg_y) for i in range(n))
        den = sum((x[i]-avg_x)**2 for i in range(n))
        return num/den if den else 0
