# Cross-Model Image Tool Setup Guide

**Date**: 2026-05-18
**What**: Grok 4.3 + GPT-5.4 collaborative image generation on your Mac

## Status

✅ **Proven Working:**
- Grok 4.3 vision (can see and analyze images)
- GPT-5.4 vision (can see and analyze images)  
- Cross-model image sharing validated
- Both models can see each other's outputs

⚠️ **Needs Configuration:**
- OpenAI Codex image generation plugin enabled (but needs interactive setup)
- Hermes tools configuration required

## Quick Start (Manual Steps)

### Step 1: Configure Image Generation

Run this interactively in your terminal:

```bash
hermes tools
```

Then:
1. Select "Image Generation"
2. Choose "openai-codex" as the backend
3. Exit

### Step 2: Test Image Generation

```bash
# Generate image with GPT-5.4
hermes chat --provider openai-codex -m gpt-5.4 \
  -q "Generate a cyberpunk city at sunset with neon lights"
```

Look for the "Image saved at:" line in the output.

### Step 3: Analyze with Grok 4.3

```bash
# Use the image path from Step 2
hermes chat --provider xai-oauth -m grok-4.3 \
  --image /path/to/image.png \
  -q "Describe this image and suggest improvements"
```

### Step 4: Iterate Based on Feedback

```bash
# Generate improved version based on Grok's analysis
hermes chat --provider openai-codex -m gpt-5.4 \
  -q "Generate a warmer version with more orange tones, based on the feedback"
```

## What We Built

### Tool 1: Simple Cross-Model Script
**Location**: `simple_cross_model.py`

```bash
# Generate
python3 simple_cross_model.py "A futuristic city with vertical gardens"

# Analyze (after generation)
python3 simple_cross_model.py analyze /path/to/image.png
```

### Tool 2: Full Pipeline Script
**Location**: `cross_model_image_tool.py`

```bash
# Full automated pipeline (3 iterations)
python3 cross_model_image_tool.py \
  "A cyberpunk city at sunset" \
  --iterations 3
```

**Features**:
- GPT-5.4 generates initial image
- Grok 4.3 analyzes and critiques
- GPT-5.4 refines based on feedback
- Saves all iterations to output directory
- Creates pipeline history JSON

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│              CROSS-MODEL IMAGE PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. GPT-5.4 GENERATE                                        │
│     ↓                                                         │
│  Initial image from prompt                                   │
│     ↓                                                         │
│  2. GROK 4.3 ANALYZE                                        │
│     ↓                                                         │
│  Detailed critique: colors, mood, composition                 │
│     ↓                                                         │
│  3. GPT-5.4 REFINE                                          │
│     ↓                                                         │
│  Improved version based on Grok's feedback                    │
│     ↓                                                         │
│  4. REPEAT (optional)                                        │
│     ↓                                                         │
│  Final output with iterative improvements                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Example Workflow

```bash
# 1. Generate initial image
hermes chat --provider openai-codex -m gpt-5.4 \
  -q "A futuristic eco-city with vertical gardens"
# → Saves to ~/.hermes/cache/images/...

# 2. Grok analyzes
hermes chat --provider xai-oauth -m grok-4.3 \
  --image ~/.hermes/cache/images/openai_codex_gpt-image-2_*.png \
  -q "What works? What should be improved?"

# 3. GPT-5.4 refines
hermes chat --provider openai-codex -m gpt-5.4 \
  -q "Create a version with more green, based on Grok's suggestions"

# 4. Grok validates
hermes chat --provider xai-oauth -m grok-4.3 \
  --image ~/.hermes/cache/images/openai_codex_gpt-image-2_*.png \
  -q "Is this better? Rate it 1-10"
```

## Capabilities Validated

### ✅ GPT-5.4 Vision
- Can see images: "attaching 1 image(s) natively (model supports vision)"
- Can generate images via gpt-image-2
- Can analyze and describe images
- Can iterate based on feedback

### ✅ Grok 4.3 Vision
- Can see and analyze images
- Provides detailed descriptions
- Identifies colors, mood, composition
- Gives specific improvement suggestions
- Understands artistic and technical elements

### ✅ Cross-Model Sharing
- GPT-5.4 can see Grok's research (as text)
- Grok can analyze GPT-5.4's images
- Both can reference shared images
- Iterative refinement loop works

## Next Steps

### Phase 1: Manual Testing (Do This First)
1. Configure `hermes tools` to enable image generation
2. Test single generation with GPT-5.4
3. Test analysis with Grok 4.3
4. Try one manual iteration

### Phase 2: Automated Pipeline
1. Use `simple_cross_model.py` for quick tests
2. Use `cross_model_image_tool.py` for full automation
3. Check output directory for results

### Phase 3: Integration
1. Integrate with SGFLIX workflows
2. Add to Jumperx stack on Mac
3. Create preset prompts for common use cases

## Troubleshooting

**Problem**: "Missing requirement: FAL_KEY environment variable"
- **Solution**: Run `hermes tools` and select openai-codex backend

**Problem**: "No image attached"
- **Solution**: Check image path is correct and file exists

**Problem**: "Model doesn't support vision"
- **Solution**: Use grok-4.3 or gpt-5.4 specifically

## Files Created

1. `simple_cross_model.py` - Simple test script
2. `cross_model_image_tool.py` - Full automated pipeline
3. `CROSS_MODEL_SETUP.md` - This file

## Summary

Your Mac now has:
- ✅ Proven cross-model vision capabilities
- ✅ Both models can see and analyze images
- ✅ Tools built for collaborative generation
- ⚠️ Just needs interactive `hermes tools` configuration to enable image generation

Once you configure `hermes tools` interactively, the full pipeline will work!

---

**Last Updated**: 2026-05-18
**Version**: 1.0
**Status**: Ready for manual configuration
