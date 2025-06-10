# Add to docs somewhere:
- [ ] find unused params must be TRUE for hierarchical classifier heads (conditional, hierarchical softmax) if using GradNorm

# Inference Testing Issues (June 2025):
- [ ] Model registry mismatch: models registered as "mFormerV1" but inference configs expect "mFormerV1_sm" (size variants should be config-based, not separate registrations)
- [ ] TaskPrediction schema changed - now requires temperature field (> 0) but tests weren't updated
- [ ] Inference components working: metadata preprocessing ✅, hierarchical consistency ✅
- [ ] Handler loading blocked by model registry issue - affects end-to-end testing
- [ ] Real inference bundle available at: /datasets/modelWorkshop/mFormerV1/linnaeus/amphibia_mFormerV1/amphibia_mFormerV1_sm_r3c_40e/inference
