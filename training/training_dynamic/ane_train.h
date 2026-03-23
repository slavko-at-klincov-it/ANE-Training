// ane_train.h — High-level ANE training and inference API
// Session-based API wrapping the ANE training pipeline.
// Compiled per model architecture: make libane_train MODEL=stories110m
//
// Training:
//   ANETrainSession *s = ane_train_create(&cfg, &err);
//   ane_train_load_data(s, "data.bin");
//   ane_train_run(s);
//   ane_train_save(s, "checkpoint.bin");
//   ane_train_destroy(s);
//
// Generation:
//   ANEGenSession *g = ane_gen_create("checkpoint.bin", "tokenizer.bin", &err);
//   ANEGenResult r = ane_gen_run(g, "Once upon a time", &cfg, on_token, NULL);
//   ane_gen_destroy(g);

#ifndef ANE_TRAIN_H
#define ANE_TRAIN_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

typedef enum {
    ANE_TRAIN_OK            = 0,   // Success
    ANE_TRAIN_ERR_ANE       = 1,   // ANE device / driver error
    ANE_TRAIN_ERR_COMPILE   = 2,   // Model compilation failed
    ANE_TRAIN_ERR_MEMORY    = 3,   // Allocation or IOSurface error
    ANE_TRAIN_ERR_CONFIG    = 4,   // Invalid configuration
    ANE_TRAIN_ERR_CHECKPOINT = 5,  // Checkpoint load/save failure
    ANE_TRAIN_ERR_DATA      = 6,   // Training data load failure
} ANETrainError;

// Returns a human-readable string for an error code.
// The returned pointer is static and must not be freed.
const char *ane_train_error_str(ANETrainError err);

// ---------------------------------------------------------------------------
// Model info
// ---------------------------------------------------------------------------

typedef struct {
    const char *name;         // Model name (e.g. "stories110m")
    int         dim;          // Embedding / hidden dimension
    int         n_layers;     // Number of transformer layers
    int         n_heads;      // Number of attention heads
    int         n_kv_heads;   // Number of key/value heads (GQA)
    int         vocab_size;   // Vocabulary size
    int         seq_len;      // Maximum sequence length
    int         hidden_dim;   // FFN intermediate dimension
    size_t      param_count;  // Total parameter count
} ANEModelInfo;

// Returns the compiled-in model dimensions.
ANEModelInfo ane_train_model_info(void);

// ---------------------------------------------------------------------------
// Hardware snapshot
// ---------------------------------------------------------------------------

typedef struct {
    float  cpu_usage;          // CPU utilisation 0–100 %
    float  gpu_usage;          // GPU utilisation 0–100 %
    size_t mem_used_bytes;     // Physical memory currently used
    size_t mem_total_bytes;    // Total physical memory
    int    thermal_state;      // 0=nominal, 1=fair, 2=serious, 3=critical
    double timestamp;          // Seconds since boot (mach_absolute_time based)
} ANEHWSnapshot;

// Captures a point-in-time hardware snapshot.
ANEHWSnapshot ane_hw_snapshot(void);

// ---------------------------------------------------------------------------
// Training — configuration
// ---------------------------------------------------------------------------

typedef struct {
    // Hyperparameters
    float  learning_rate;      // Peak learning rate (default: 5e-4)
    float  min_lr;             // Minimum learning rate after decay
    int    warmup_steps;       // Linear warmup steps
    int    max_steps;          // Total training steps (0 = unlimited)
    int    batch_size;         // Micro-batch size
    int    accum_steps;        // Gradient accumulation steps (>= 100)

    // Adam
    float  beta1;              // Adam beta1 (default: 0.9)
    float  beta2;              // Adam beta2 (default: 0.999)
    float  eps;                // Adam epsilon (default: 1e-8)
    float  weight_decay;       // AdamW weight decay (default: 0.01)

    // Numeric stability
    float  loss_scale;         // FP16 loss scaling (default: 1024.0)
    float  grad_clip;          // Max gradient norm, 0 = no clipping

    // Checkpointing
    int    checkpoint_every;   // Save checkpoint every N steps (0 = never)
    const char *checkpoint_dir; // Directory for auto-checkpoints (NULL = cwd)

    // Logging
    int    log_every;          // Print stats every N steps (0 = never)
} ANETrainConfig;

// Returns a config initialised with sane defaults.
ANETrainConfig ane_train_default_config(void);

// ---------------------------------------------------------------------------
// Training — per-step result
// ---------------------------------------------------------------------------

typedef struct {
    int    step;               // Current step number (1-based)
    float  loss;               // Cross-entropy loss this step
    float  lr;                 // Learning rate used this step
    float  grad_norm;          // Global gradient L2 norm (pre-clip)

    // Timing breakdown (milliseconds)
    double fwd_ms;             // Forward pass on ANE
    double bwd_ms;             // Backward pass on ANE
    double update_ms;          // Weight update (Adam on CPU)
    double total_ms;           // Wall-clock time for entire step
} ANETrainStepResult;

// ---------------------------------------------------------------------------
// Training — cumulative stats
// ---------------------------------------------------------------------------

typedef struct {
    int    steps_done;         // Total steps completed
    float  best_loss;          // Lowest loss observed
    int    best_loss_step;     // Step at which best_loss occurred
    double elapsed_s;          // Total wall-clock seconds
    double avg_step_ms;        // Average milliseconds per step
    double compile_ms;         // One-time compilation time
    float  tflops;             // Estimated sustained TFLOPS
} ANETrainStats;

// ---------------------------------------------------------------------------
// Training — callbacks
// ---------------------------------------------------------------------------

// Called after each step. Return false to stop training early.
typedef bool (*ANETrainProgressFn)(const ANETrainStepResult *result, void *ctx);

// Called when a checkpoint is about to be saved.
// path is the target file path. Return false to skip this checkpoint.
typedef bool (*ANETrainCheckpointFn)(int step, const char *path, void *ctx);

// ---------------------------------------------------------------------------
// Training — session (opaque)
// ---------------------------------------------------------------------------

typedef struct ANETrainSession ANETrainSession;

// Creates a training session. Compiles the model on first call.
// On failure returns NULL and sets *err (if err is not NULL).
ANETrainSession *ane_train_create(const ANETrainConfig *cfg, ANETrainError *err);

// Loads training data from a binary token file.
// Must be called before ane_train_step / ane_train_run.
ANETrainError ane_train_load_data(ANETrainSession *s, const char *path);

// Resumes training from a checkpoint file.
// Restores weights, optimizer state, and step counter.
ANETrainError ane_train_resume(ANETrainSession *s, const char *checkpoint_path);

// Registers a progress callback (called after every step).
void ane_train_set_progress_callback(ANETrainSession *s,
                                     ANETrainProgressFn fn, void *ctx);

// Registers a checkpoint callback (called before each auto-save).
void ane_train_set_checkpoint_callback(ANETrainSession *s,
                                       ANETrainCheckpointFn fn, void *ctx);

// Runs a single training step. Returns the step result.
ANETrainStepResult ane_train_step(ANETrainSession *s);

// Runs training for cfg.max_steps (or until callback returns false).
ANETrainError ane_train_run(ANETrainSession *s);

// Returns cumulative training statistics.
ANETrainStats ane_train_stats(const ANETrainSession *s);

// Saves a checkpoint to the given path.
ANETrainError ane_train_save(const ANETrainSession *s, const char *path);

// Destroys the session and frees all resources.
void ane_train_destroy(ANETrainSession *s);

// ---------------------------------------------------------------------------
// Generation — configuration
// ---------------------------------------------------------------------------

typedef struct {
    float    temperature;      // Sampling temperature (default: 0.8)
    float    top_p;            // Nucleus sampling threshold (default: 0.9)
    int      max_tokens;       // Maximum tokens to generate (default: 256)
    uint64_t seed;             // RNG seed (0 = random)
} ANEGenConfig;

// Returns a generation config with sane defaults.
ANEGenConfig ane_gen_default_config(void);

// ---------------------------------------------------------------------------
// Generation — streaming callback
// ---------------------------------------------------------------------------

// Called for each generated token. Return false to stop generation early.
// token is the decoded text piece (UTF-8, null-terminated).
typedef bool (*ANEGenTokenFn)(const char *token, int token_id, void *ctx);

// ---------------------------------------------------------------------------
// Generation — result
// ---------------------------------------------------------------------------

typedef enum {
    ANE_GEN_STOP_EOS       = 0,   // End-of-sequence token reached
    ANE_GEN_STOP_MAX       = 1,   // max_tokens reached
    ANE_GEN_STOP_CALLBACK  = 2,   // Callback returned false
    ANE_GEN_STOP_ERROR     = 3,   // Error during generation
} ANEGenStopReason;

typedef struct {
    char          *text;             // Full generated text (caller must free via ane_gen_result_free)
    int            tokens_generated; // Number of tokens produced
    double         total_ms;         // Total generation wall-clock time
    double         ms_per_token;     // Average latency per token
    ANEGenStopReason stop_reason;    // Why generation stopped
} ANEGenResult;

// ---------------------------------------------------------------------------
// Generation — session (opaque)
// ---------------------------------------------------------------------------

typedef struct ANEGenSession ANEGenSession;

// Creates a generation session from a checkpoint and tokenizer.
// On failure returns NULL and sets *err (if err is not NULL).
ANEGenSession *ane_gen_create(const char *checkpoint_path,
                              const char *tokenizer_path,
                              ANETrainError *err);

// Generates text from a prompt. The callback (if non-NULL) streams tokens.
// Returns a result struct; caller must call ane_gen_result_free when done.
ANEGenResult ane_gen_run(ANEGenSession *g, const char *prompt,
                         const ANEGenConfig *cfg,
                         ANEGenTokenFn on_token, void *ctx);

// Frees the text buffer inside a generation result.
void ane_gen_result_free(ANEGenResult *r);

// Destroys the generation session and frees all resources.
void ane_gen_destroy(ANEGenSession *g);

#ifdef __cplusplus
}
#endif

#endif // ANE_TRAIN_H
