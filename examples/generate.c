// generate.c — ANE Text Generation
// Trains a bigram model on Shakespeare, generates text char-by-char with typewriter effect
//
// Build & run: make generate

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "../libane/ane.h"

// ===== Vocabulary =====
#define VOCAB 64
#define SEQ   32

// Character mapping: a-z (26) + A-Z (26) + 0-9 (digits map to 52-61) + space, newline, '.', ',' = 64
static char idx_to_char[VOCAB];
static int  char_to_idx[256];

static void init_vocab(void) {
    memset(char_to_idx, -1, sizeof(char_to_idx));
    int idx = 0;
    for (int c = 'a'; c <= 'z'; c++) { idx_to_char[idx] = c; char_to_idx[c] = idx++; }
    for (int c = 'A'; c <= 'Z'; c++) { idx_to_char[idx] = c; char_to_idx[c] = idx++; }
    // 52-55: space, newline, period, comma
    idx_to_char[52] = ' ';  char_to_idx[(int)' ']  = 52;
    idx_to_char[53] = '\n'; char_to_idx[(int)'\n'] = 53;
    idx_to_char[54] = '.';  char_to_idx[(int)'.']  = 54;
    idx_to_char[55] = ',';  char_to_idx[(int)',']  = 55;
    idx_to_char[56] = '!';  char_to_idx[(int)'!']  = 56;
    idx_to_char[57] = '?';  char_to_idx[(int)'?']  = 57;
    idx_to_char[58] = ':';  char_to_idx[(int)':']  = 58;
    idx_to_char[59] = ';';  char_to_idx[(int)';']  = 59;
    idx_to_char[60] = '\''; char_to_idx[(int)'\''] = 60;
    idx_to_char[61] = '-';  char_to_idx[(int)'-']  = 61;
    idx_to_char[62] = '(';  char_to_idx[(int)'(']  = 62;
    idx_to_char[63] = ')';  char_to_idx[(int)')']  = 63;
}

// ===== Shakespeare training data =====
static const char *SHAKESPEARE =
    "To be, or not to be, that is the question.\n"
    "Whether tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles,\n"
    "And by opposing end them. To die, to sleep.\n"
    "No more, and by a sleep to say we end\n"
    "The heartache and the thousand natural shocks\n"
    "That flesh is heir to. Tis a consummation\n"
    "Devoutly to be wished. To die, to sleep,\n"
    "To sleep, perchance to dream. Ay, there's the rub,\n"
    "For in that sleep of death what dreams may come.\n";

// ===== Softmax =====
static void softmax(float *x, int n) {
    // Clamp to FP16 safe range before softmax
    for (int i = 0; i < n; i++)
        x[i] = fminf(fmaxf(x[i], -65504.0f), 65504.0f);
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ===== Temperature sampling =====
static int sample(float *probs, int n, float temperature) {
    // Apply temperature
    float logits[VOCAB];
    for (int i = 0; i < n; i++) logits[i] = logf(probs[i] + 1e-10f) / temperature;
    softmax(logits, n);

    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    for (int i = 0; i < n; i++) {
        cum += logits[i];
        if (r <= cum) return i;
    }
    return n - 1;
}

int main(void) {
    srand((unsigned)time(NULL));
    init_vocab();

    printf("\n");
    printf("  \xe2\x9c\x8d  ANE TEXT GENERATOR\n");
    printf("\n");

    if (ane_init() != 0) { printf("  ERROR: ANE init failed.\n"); return 1; }
    ANEDeviceInfo info = ane_device_info();
    printf("  Hardware: %s, %d cores\n", info.arch ? info.arch : "?", info.num_cores);

    // ===== Encode training data =====
    int data_len = (int)strlen(SHAKESPEARE);
    int *encoded = (int *)malloc(data_len * sizeof(int));
    int valid = 0;
    for (int i = 0; i < data_len; i++) {
        int idx = char_to_idx[(unsigned char)SHAKESPEARE[i]];
        if (idx >= 0) encoded[valid++] = idx;
    }
    printf("  Training data: %d characters encoded\n", valid);

    // ===== Build bigram training pairs =====
    // For each consecutive pair (c1, c2): input is one-hot(c1), target is one-hot(c2)
    // We batch SEQ pairs per training step
    int n_pairs = valid - 1;
    printf("  Training pairs: %d bigrams\n", n_pairs);

    // ===== Weight matrix W[VOCAB x VOCAB] =====
    float W[VOCAB * VOCAB];
    for (int i = 0; i < VOCAB * VOCAB; i++) W[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    float bias[VOCAB];
    memset(bias, 0, sizeof(bias));

    size_t io_bytes = (size_t)VOCAB * SEQ * 4;

    // ===== Training =====
    printf("\n  Training bigram model on Shakespeare...\n\n");
    printf("  step   loss      perplexity\n");
    printf("  ----   --------  ----------\n");

    int steps = 30;
    float lr = 0.5f;

    for (int step = 0; step < steps; step++) {
        // Build batch: SEQ random bigram pairs
        float input[VOCAB * SEQ];
        int targets[SEQ];
        memset(input, 0, sizeof(input));

        for (int s = 0; s < SEQ; s++) {
            int pair_idx = rand() % n_pairs;
            int c_in = encoded[pair_idx];
            int c_out = encoded[pair_idx + 1];
            input[c_in * SEQ + s] = 1.0f;  // one-hot
            targets[s] = c_out;
        }

        // Forward on ANE
        ANEWeight w = ane_weight_fp16("@model_path/weights/weight.bin", W, VOCAB, VOCAB);
        char *mil = ane_mil_linear(VOCAB, VOCAB, SEQ, "@model_path/weights/weight.bin");
        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                    1, &io_bytes, 1, &io_bytes,
                                    ANE_QOS_BACKGROUND);
        if (!k) {
            printf("  Compile failed at step %d (budget: %d/%d)\n",
                   step, ane_compile_count(), ANE_COMPILE_BUDGET);
            free(mil); ane_weight_free(&w);
            break;
        }

        ane_write(k, 0, input, io_bytes);
        ane_eval(k, ANE_QOS_BACKGROUND);

        float output[VOCAB * SEQ];
        ane_read(k, 0, output, io_bytes);

        // Sanitize ANE output
        for (int i = 0; i < VOCAB * SEQ; i++) {
            if (isnan(output[i]) || isinf(output[i])) output[i] = 0.0f;
        }

        // Cross-entropy loss + gradient
        float loss = 0;
        float grad_W[VOCAB * VOCAB];
        memset(grad_W, 0, sizeof(grad_W));

        for (int s = 0; s < SEQ; s++) {
            // Add bias and softmax for this position
            float logits[VOCAB];
            for (int v = 0; v < VOCAB; v++) logits[v] = output[v * SEQ + s] + bias[v];
            softmax(logits, VOCAB);

            // CE loss
            float p = logits[targets[s]];
            if (p < 1e-7f) p = 1e-7f;
            loss -= logf(p);

            // Gradient: d_logits = probs - one_hot(target)
            float d_logits[VOCAB];
            for (int v = 0; v < VOCAB; v++) d_logits[v] = logits[v];
            d_logits[targets[s]] -= 1.0f;

            // dW[i][j] += d_logits[i] * input[j][s]
            for (int i = 0; i < VOCAB; i++) {
                for (int j = 0; j < VOCAB; j++) {
                    grad_W[i * VOCAB + j] += d_logits[i] * input[j * SEQ + s];
                }
                bias[i] -= lr * d_logits[i] / SEQ;
            }
        }
        loss /= SEQ;

        if (isnan(loss)) { printf("  NaN loss at step %d, stopping\n", step); break; }

        // Sanitize gradients
        for (int i = 0; i < VOCAB * VOCAB; i++) {
            if (isnan(grad_W[i])) grad_W[i] = 0.0f;
            if (isinf(grad_W[i])) grad_W[i] = copysignf(65504.0f, grad_W[i]);
        }

        // SGD update
        for (int i = 0; i < VOCAB * VOCAB; i++) W[i] -= lr * grad_W[i] / SEQ;

        if (step < 3 || step % 5 == 0 || step == steps - 1)
            printf("  %-4d   %8.4f  %8.2f\n", step, loss, expf(loss));

        ane_free(k);
        free(mil);
        ane_weight_free(&w);
    }

    // ===== Generation =====
    printf("\n  Generating text (200 chars, temperature=0.8)...\n");
    printf("  -------------------------------------------\n  ");

    // Compile generation kernel with final trained weights
    ANEWeight w_gen = ane_weight_fp16("@model_path/weights/weight.bin", W, VOCAB, VOCAB);
    char *mil_gen = ane_mil_linear(VOCAB, VOCAB, SEQ, "@model_path/weights/weight.bin");
    ANEKernel *gen_k = ane_compile(mil_gen, strlen(mil_gen), &w_gen, 1,
                                    1, &io_bytes, 1, &io_bytes,
                                    ANE_QOS_BACKGROUND);

    if (!gen_k) {
        printf("\n  Could not compile generation kernel (budget: %d/%d)\n",
               ane_compile_count(), ANE_COMPILE_BUDGET);
        free(mil_gen);
        ane_weight_free(&w_gen);
        free(encoded);
        return 0;
    }

    // Start with a random character
    int current = char_to_idx[(int)'T'];
    float temperature = 0.8f;

    for (int g = 0; g < 200; g++) {
        // One-hot input for current char (use position 0 of SEQ)
        float gen_input[VOCAB * SEQ];
        memset(gen_input, 0, sizeof(gen_input));
        gen_input[current * SEQ + 0] = 1.0f;

        ane_write(gen_k, 0, gen_input, io_bytes);
        ane_eval(gen_k, ANE_QOS_BACKGROUND);

        float gen_output[VOCAB * SEQ];
        ane_read(gen_k, 0, gen_output, io_bytes);

        // Get logits for position 0
        float probs[VOCAB];
        for (int v = 0; v < VOCAB; v++) probs[v] = gen_output[v * SEQ + 0] + bias[v];
        softmax(probs, VOCAB);

        current = sample(probs, VOCAB, temperature);

        char c = idx_to_char[current];
        putchar(c);
        fflush(stdout);
        usleep(30000);  // 30ms typewriter effect
    }

    printf("\n  -------------------------------------------\n");

    ane_free(gen_k);
    free(mil_gen);
    ane_weight_free(&w_gen);
    free(encoded);

    printf("\n  Compiles used: %d / %d\n\n", ane_compile_count(), ANE_COMPILE_BUDGET);
    return 0;
}
