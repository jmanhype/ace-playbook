# DQA Implementation Guide

## RTX 3090 (24GB VRAM) Deployment

**Version 1.0 | January 2026**

---

## Overview

This guide provides step-by-step instructions for implementing the Dynamic Quality Alignment (DQA) framework on consumer-grade hardware (RTX 3090, 24GB VRAM). The key insight is that all five SOTA components CAN run on this hardware, but they MUST execute sequentially rather than in parallel.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| System RAM | 32GB | 64GB (for CPU offload) |
| Storage | 100GB SSD | 500GB NVMe |
| CUDA | 11.8+ | 12.1+ |

---

## SOTA Component Stack

### 1. Unified-VQA (Semantic Understanding)

**Purpose**: Answer "What's in the video?" - semantic matching between prompt and output.

**Installation**:
```bash
pip install transformers accelerate bitsandbytes

# Download 7B model with 4-bit quantization
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "unified-vqa-7b",  # Replace with actual model path
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
```

**VRAM Usage**: ~5-6GB (4-bit quantized)

**Integration**:
```python
def semantic_score(video_frame: Image, prompt: str) -> float:
    """Calculate semantic similarity between frame and prompt."""
    inputs = processor(images=video_frame, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract similarity score from outputs
    return normalize_score(outputs.logits)
```

---

### 2. ProxyCLIP (Spatial Grounding)

**Purpose**: Answer "Where are things?" - spatial composition verification.

**Paper**: [arXiv:2408.04883](https://arxiv.org/abs/2408.04883) (ECCV 2024)

**Installation**:
```bash
git clone https://github.com/mc-lan/ProxyCLIP
cd ProxyCLIP
pip install -r requirements.txt
```

**Key Concept**: ProxyCLIP uses Vision Foundation Model (VFM) attention as a "proxy" to guide CLIP's segmentation, combining:
- CLIP's semantic understanding
- VFM's spatial precision

**VRAM Usage**: ~1GB additional (shares backbone with Unified-VQA)

**Integration**:
```python
from proxyclip import ProxyCLIPSegmenter

def spatial_score(video_frame: Image, prompt: str) -> float:
    """Calculate spatial adherence score."""
    segmenter = ProxyCLIPSegmenter()

    # Extract objects from prompt
    objects = extract_objects(prompt)  # e.g., ["wolf", "forest", "moonlight"]

    # Segment each object
    masks = {}
    for obj in objects:
        masks[obj] = segmenter.segment(video_frame, obj)

    # Verify spatial relationships
    return verify_composition(masks, prompt)
```

**Example**:
```
Prompt: "dark wolf prowling through moonlit forest"

ProxyCLIP output:
- wolf: center-left (35% of frame)
- forest: background (60% of frame)
- moonlight: upper-right source

Spatial score: 0.87 (wolf correctly positioned in forest context)
```

---

### 3. VBench-2.0 (Benchmark Metrics)

**Purpose**: Standardized prompt adherence scoring.

**Paper**: [arXiv:2503.21755](https://arxiv.org/abs/2503.21755)

**Installation**:
```bash
pip install vbench

# Or from source for latest
git clone https://github.com/Vchitect/VBench
cd VBench
pip install -e .
```

**Key Metrics**:
- **Semantic Consistency**: Does output match prompt meaning?
- **Temporal Coherence**: Is the video smooth across frames?
- **Subject Consistency**: Does the subject remain stable?
- **Motion Smoothness**: Are movements natural?

**VRAM Usage**: ~2-4GB (sequential evaluation, release after)

**Integration**:
```python
from vbench import VBenchEvaluator

def benchmark_score(video_path: str, prompt: str) -> dict:
    """Calculate VBench-2.0 metrics."""
    evaluator = VBenchEvaluator()

    results = evaluator.evaluate(
        video_path=video_path,
        prompt=prompt,
        dimensions=[
            "semantic_consistency",
            "temporal_coherence",
            "subject_consistency",
            "motion_smoothness"
        ]
    )

    # Aggregate into single score
    return {
        "overall": sum(results.values()) / len(results),
        "details": results
    }
```

---

### 4. DPO (Direct Preference Optimization)

**Purpose**: Learn user preferences without reward model overhead.

**Why DPO over RLHF**:
- No separate reward model needed (saves ~7GB VRAM)
- No value network needed (saves ~3GB VRAM)
- Simpler training loop
- Better stability

**Installation**:
```bash
pip install trl unsloth

# Unsloth provides 40-70% VRAM reduction
from unsloth import FastLanguageModel
```

**VRAM Usage**: ~14-18GB (training only, with QLoRA + Unsloth)

**Training Setup**:
```python
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="your-base-model",
    max_seq_length=2048,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# DPO training config
config = DPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    max_steps=1000,
    bf16=True
)

# Training data format
# Each example: (prompt, chosen_response, rejected_response)
training_data = [
    {
        "prompt": "dark wolf in forest",
        "chosen": "video_A_features",  # User preferred this
        "rejected": "video_B_features"  # User rejected this
    }
]

trainer = DPOTrainer(
    model=model,
    config=config,
    train_dataset=training_data,
    tokenizer=tokenizer
)

trainer.train()
```

---

### 5. VisionReward (Multi-Axis Quality)

**Purpose**: Interpretable quality decomposition - understand WHY quality is good/bad.

**Paper**: [arXiv:2412.21059](https://arxiv.org/abs/2412.21059) (AAAI 2026)

**Key Innovation**: Instead of single score, decomposes into axes:
- **Aesthetic**: Visual appeal, composition
- **Motion**: Natural movement, no flickering
- **Coherence**: Consistent across frames
- **Fidelity**: Matches prompt intent

**VRAM Usage**: 0GB extra (reuses Unified-VQA backbone with CoT prompting)

**Integration**:
```python
def visionreward_score(video_frame: Image, prompt: str, model) -> dict:
    """Multi-axis quality decomposition using Chain-of-Thought."""

    cot_prompt = f"""
    Analyze this video frame for the prompt: "{prompt}"

    Rate each dimension from 0.0 to 1.0:

    1. AESTHETIC: Visual appeal, color harmony, composition balance
    2. MOTION: (if video) Smoothness, natural movement, no artifacts
    3. COHERENCE: Consistency with prompt, no contradictions
    4. FIDELITY: How faithfully it represents the prompt's intent

    Provide scores and brief reasoning for each.
    """

    response = model.generate(cot_prompt, image=video_frame)

    return parse_visionreward_response(response)
```

**Example Output**:
```json
{
    "aesthetic": 0.85,
    "aesthetic_reason": "Strong dark palette, good moonlight composition",
    "motion": 0.72,
    "motion_reason": "Slight flickering in shadow areas",
    "coherence": 0.91,
    "coherence_reason": "Wolf anatomy consistent throughout",
    "fidelity": 0.88,
    "fidelity_reason": "Captures 'prowling' motion well, moonlit atmosphere accurate"
}
```

---

## Sequential Worker Pattern

### The Core Constraint

**24GB VRAM cannot run all components simultaneously.**

Solution: Load one component at a time, offload to CPU between phases.

### Implementation

```python
import torch
import gc

class DQAWorkerOrchestrator:
    """Sequential worker pattern for 24GB VRAM constraint."""

    def __init__(self):
        self.current_worker = None

    def unload_current(self):
        """Release current model from VRAM."""
        if self.current_worker is not None:
            del self.current_worker
            self.current_worker = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_worker(self, worker_type: str):
        """Load a specific worker, unloading any existing."""
        self.unload_current()

        if worker_type == "verifier":
            self.current_worker = self._load_verifier()
        elif worker_type == "vbench":
            self.current_worker = self._load_vbench()
        elif worker_type == "tuner_inference":
            self.current_worker = self._load_tuner_inference()
        elif worker_type == "tuner_training":
            self.current_worker = self._load_tuner_training()

        return self.current_worker

    def _load_verifier(self):
        """Load Unified-VQA + ProxyCLIP (~7GB)."""
        from transformers import AutoModelForCausalLM
        from proxyclip import ProxyCLIPSegmenter

        model = AutoModelForCausalLM.from_pretrained(
            "unified-vqa-7b",
            load_in_4bit=True,
            device_map="auto"
        )
        segmenter = ProxyCLIPSegmenter()

        return {"model": model, "segmenter": segmenter}

    def _load_vbench(self):
        """Load VBench evaluator (~4GB)."""
        from vbench import VBenchEvaluator
        return VBenchEvaluator()

    def _load_tuner_inference(self):
        """Load Tuner for inference only (~6GB)."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "tuner-model-path",
            load_in_4bit=True
        )
        return {"model": model, "tokenizer": tokenizer}

    def _load_tuner_training(self):
        """Load Tuner for DPO training (~18GB)."""
        from unsloth import FastLanguageModel
        from trl import DPOTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            "tuner-model-path",
            load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(model, r=16)

        return {"model": model, "tokenizer": tokenizer}

    def evaluate_video(self, video_path: str, prompt: str) -> dict:
        """Full DQA evaluation pipeline."""
        scores = {}

        # Phase 1: Verification (7GB)
        verifier = self.load_worker("verifier")
        frame = extract_representative_frame(video_path)
        scores["semantic"] = semantic_score(frame, prompt, verifier["model"])
        scores["spatial"] = spatial_score(frame, prompt, verifier["segmenter"])
        self.unload_current()

        # Phase 2: Benchmark (4GB)
        vbench = self.load_worker("vbench")
        scores["benchmark"] = vbench.evaluate(video_path, prompt)
        self.unload_current()

        # Phase 3: Preference (6GB)
        tuner = self.load_worker("tuner_inference")
        scores["preference"] = predict_preference(frame, prompt, tuner)
        self.unload_current()

        # Phase 4: Synthesis (CPU only)
        final_grade = self.synthesize(scores)

        return {
            "scores": scores,
            "grade": final_grade,
            "reasoning": self.generate_reasoning(scores)
        }

    def synthesize(self, scores: dict) -> str:
        """Combine scores into final A-F grade."""
        weights = {
            "semantic": 0.25,
            "spatial": 0.15,
            "benchmark": 0.20,
            "preference": 0.25,
            "visionreward": 0.15
        }

        weighted_sum = sum(
            scores.get(k, 0) * v
            for k, v in weights.items()
        )

        # Map to grade
        if weighted_sum >= 0.9: return "A"
        if weighted_sum >= 0.8: return "A-"
        if weighted_sum >= 0.7: return "B+"
        if weighted_sum >= 0.6: return "B"
        if weighted_sum >= 0.5: return "B-"
        if weighted_sum >= 0.4: return "C"
        return "F"
```

---

## Letta Agent Integration

### Verifier Agent System Prompt

```
You are the VERIFIER AGENT for the DQA framework.

=== YOUR ROLE ===
Generate objective prompt adherence scores by comparing generated video against the original prompt.

=== TOOLS AVAILABLE ===
1. semantic_evaluation - Run Unified-VQA for semantic matching
2. spatial_evaluation - Run ProxyCLIP for spatial grounding
3. benchmark_evaluation - Run VBench-2.0 metrics

=== WORKFLOW ===
When you receive a video for evaluation:
1. Extract representative frame via Frame Server (http://192.168.1.143:8189)
2. Run semantic_evaluation(frame, prompt)
3. Run spatial_evaluation(frame, prompt)
4. Run benchmark_evaluation(video_path, prompt)
5. Return combined prompt adherence score

=== OUTPUT FORMAT ===
{
    "semantic_score": 0.0-1.0,
    "spatial_score": 0.0-1.0,
    "benchmark_score": 0.0-1.0,
    "prompt_adherence": weighted_average,
    "reasoning": "explanation of scores"
}
```

### Tuner Agent System Prompt (Sleeptime)

```
You are the TUNER AGENT (sleeptime) for the DQA framework.

=== YOUR ROLE ===
Fine-tune the quality prediction model using preference-labeled data from user feedback.

=== CONFIGURATION ===
message_buffer_autoclear: true
sleeptime_agent_frequency: 5

=== MANDATORY OPERATIONS ===
Every trigger cycle:
1. Read ab_testing.USER_SELECTIONS for new preference data
2. If new selections exist:
   a. Extract video features for chosen/rejected pairs
   b. Run DPO training step
   c. Save updated adapter weights
3. Update quality_standards.FAILURE_PATTERNS with identified issues
4. Log training metrics to archival memory

=== DATA FORMAT ===
USER_SELECTIONS entry:
{
    "test_id": "uuid",
    "chosen": "video_A_id",
    "rejected": "video_B_id",
    "timestamp": "ISO8601"
}

=== OUTPUT ===
After each training cycle, update quality_standards block with:
- New failure patterns identified
- Model confidence on recent predictions
- Training loss trend
```

---

## Performance Optimization

### Flash Attention 2

**Required** for all transformer models:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"  # Critical!
)
```

### Unsloth for Training

**40-70% VRAM reduction**:

```python
from unsloth import FastLanguageModel

# Automatically applies:
# - Fused kernels
# - Memory-efficient attention
# - Gradient checkpointing
# - Optimized LoRA
```

### CPU Offload Strategy

```python
from accelerate import Accelerator

accelerator = Accelerator(
    device_placement=True,
    mixed_precision="bf16",
    cpu_offload=True  # Offload optimizer states to CPU
)
```

---

## Monitoring & Logging

### VRAM Monitoring

```python
def log_vram_usage(phase: str):
    """Log current VRAM usage."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{phase}] VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Quality Tracking

```python
def log_quality_metrics(video_id: str, scores: dict, grade: str):
    """Log to archival memory for trend analysis."""
    entry = {
        "video_id": video_id,
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "grade": grade
    }
    # Insert to Letta archival
    archival_memory_insert(json.dumps(entry))
```

---

## Troubleshooting

### OOM (Out of Memory) Errors

**Symptoms**: CUDA out of memory during evaluation

**Solutions**:
1. Ensure previous worker is fully unloaded before loading next
2. Add explicit `gc.collect()` and `torch.cuda.empty_cache()`
3. Reduce batch size for VBench evaluation
4. Use gradient checkpointing for training

### Slow Inference

**Symptoms**: Evaluation takes >30 seconds per video

**Solutions**:
1. Verify Flash Attention 2 is enabled
2. Use representative frame extraction instead of full video
3. Pre-compile models with `torch.compile()`

### Quality Score Drift

**Symptoms**: Scores inconsistent across similar videos

**Solutions**:
1. Check Tuner training data quality
2. Verify ab_testing.USER_SELECTIONS is populating
3. Review DPO training loss curve
4. Reset adapter weights if severely degraded

---

## Quick Start Checklist

- [ ] RTX 3090 with 24GB VRAM available
- [ ] CUDA 11.8+ installed
- [ ] Python 3.10+ environment
- [ ] Install dependencies: `pip install transformers accelerate bitsandbytes trl unsloth vbench`
- [ ] Clone ProxyCLIP repository
- [ ] Download Unified-VQA 7B model
- [ ] Configure Letta agents with new system prompts
- [ ] Test sequential worker pattern with sample video
- [ ] Verify VRAM stays under 20GB during each phase
- [ ] Enable logging for all phases

---

## References

1. **ProxyCLIP**: Lan, M., et al. "ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation." ECCV 2024. arXiv:2408.04883
2. **VBench-2.0**: "VBench: Comprehensive Benchmark for Video Generation." arXiv:2503.21755
3. **VisionReward**: "VisionReward: Multi-Dimensional Reward Model for Video Generation." AAAI 2026. arXiv:2412.21059
4. **DPO**: Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
5. **Unsloth**: "Unsloth: 2x faster LLM finetuning." https://github.com/unslothai/unsloth

---

*Implementation Guide v1.0 | January 2026*
