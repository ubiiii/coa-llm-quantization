# ==========================================================
# GPT-2 ONNX Colab Sampler — Complete Optimized Implementation
# ==========================================================
# Professional-grade ONNX Runtime inference with advanced text generation
# Fixes for function-word attractors and INT8 quantization issues

import os, time, numpy as np
from collections import defaultdict, deque
import onnxruntime as ort
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class ONNXGPT2Sampler:
    """
    Professional ONNX Runtime GPT-2 sampler with advanced text generation controls.
    
    Features:
    - INT8 quantization support with optimized parameters
    - Function-word attractor prevention
    - Advanced loop detection and prevention
    - KV cache optimization for autoregressive generation
    - Comprehensive safety nets and fallbacks
    """
    
    def __init__(self, model_path=None, use_cuda=False):
        """
        Initialize the ONNX GPT-2 sampler.
        
        Args:
            model_path: Path to ONNX model (auto-detects if None)
            use_cuda: Whether to use CUDA providers
        """
        self.model_path = model_path or self._find_best_model()
        self.use_cuda = use_cuda
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        
        # Precompute problematic tokens for efficiency
        self.function_word_tokens = self._get_function_word_tokens()
        self.short_tokens = self._get_short_tokens()
        self.punctuation_tokens = self._get_punctuation_tokens()
        
        # Initialize session and schema
        self.session = self._build_session()
        self.schema = self._io_schema()
        
        # Initialize loop detector
        self.loop_detector = self._create_loop_detector()
        
        print(f"✅ Initialized ONNX GPT-2 Sampler with model: {self.model_path}")
    
    def _find_best_model(self):
        """Find the best available ONNX model."""
        candidates = [
            "/content/model.with_past.onnx",      # FP32 with KV cache
            "/content/model.with_past.int8.onnx", # INT8 with KV cache
            "/content/model.int8.onnx",           # INT8 basic
            "/content/model.onnx",                # FP32 basic
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("No GPT-2 ONNX model found in /content")
    
    def _get_function_word_tokens(self):
        """Precompute function words that cause attractors."""
        function_words = [
            " and", " And", " the", " The", " do", " Do", " of", " to", 
            " a", " A", " is", " are", " was", " were", " in", " on", 
            " at", " by", " for", " with", " from", " up", " down"
        ]
        function_tokens = set()
        for word in function_words:
            try:
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                function_tokens.update(tokens)
            except:
                pass
        return function_tokens
    
    def _get_short_tokens(self):
        """Precompute short tokens (≤2 characters)."""
        short_tokens = set()
        for token_id in range(len(self.tokenizer.vocab)):
            try:
                token_str = self.tokenizer.decode([token_id])
                if len(token_str.strip()) <= 2:
                    short_tokens.add(token_id)
            except:
                pass
        return short_tokens
    
    def _get_punctuation_tokens(self):
        """Precompute punctuation tokens for shaping."""
        punctuation = [".", ",", ";", ":", "!", "?"]
        punct_tokens = set()
        for punct in punctuation:
            try:
                tokens = self.tokenizer.encode(punct, add_special_tokens=False)
                punct_tokens.update(tokens)
            except:
                pass
        return punct_tokens
    
    def _create_loop_detector(self):
        """Create enhanced loop detector."""
        return EnhancedLoopDetector()
    
    def _build_session(self):
        """Build ONNX Runtime session with optimizations."""
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        if self.use_cuda:
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0, "do_copy_in_default_stream": 1}),
                "CPUExecutionProvider"
            ]
        else:
            providers = ["CPUExecutionProvider"]
        
        return ort.InferenceSession(self.model_path, sess_options=so, providers=providers)
    
    def _io_schema(self):
        """Extract input/output schema from ONNX model."""
        inps = self.session.get_inputs()
        outs = self.session.get_outputs()
        in_names = [i.name for i in inps]
        out_names = [o.name for o in outs]
        
        kv_inputs = [n for n in in_names if ("past_key" in n or "past_value" in n or n.startswith("past"))]
        kv_outputs = [n for n in out_names if ("present" in n or "past_key_values" in n or "present_key" in n)]
        
        schema = {
            "input_ids": next((n for n in in_names if n.endswith("input_ids") or n == "input_ids"), None),
            "attention_mask": next((n for n in in_names if n.endswith("attention_mask") or n == "attention_mask"), None),
            "kv_inputs": sorted(kv_inputs),
            "logits_out": next((n for n in out_names if n.endswith("logits") or n == "logits"), out_names[0]),
            "kv_outputs": sorted(kv_outputs),
            "input_meta": {i.name: i for i in inps}
        }
        schema["has_kv"] = (len(schema["kv_inputs"]) == len(schema["kv_outputs"]) > 0)
        schema["kv_required"] = len(schema["kv_inputs"]) > 0
        
        return schema
    
    def encode(self, text):
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)
    
    def generate(self, prompt, max_new_tokens=64, temperature=1.10, top_k=70, top_p=0.95,
                rep_penalty=1.24, freq_lambda=0.62, pres_lambda=0.22, ngram_block=4,
                seed=42):
        """
        Generate text using optimized ONNX Runtime inference.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.10 optimal for INT8)
            top_k: Top-k sampling (70 recommended)
            top_p: Top-p sampling (0.95 recommended)
            rep_penalty: Repetition penalty (1.24 recommended)
            freq_lambda: Frequency penalty (0.62 recommended)
            pres_lambda: Presence penalty (0.22 recommended)
            ngram_block: N-gram blocking (4 recommended)
            seed: Random seed for reproducibility
        
        Returns:
            Generated text continuation
        """
        # Initialize sampling
        rng = np.random.default_rng(seed)
        loop_detector = EnhancedLoopDetector()
        
        # Encode prompt
        prompt_ids = self.encode(prompt)
        generated_ids = prompt_ids.copy()
        
        # Initialize past cache
        past = None
        seq_pos = len(prompt_ids)
        
        # Generation loop
        for step in range(max_new_tokens):
            # Get logits for last token
            if len(generated_ids) == len(prompt_ids):
                # First step: use full prompt
                input_ids = np.array([generated_ids], dtype=np.int64)
                feeds = {self.schema["input_ids"]: input_ids}
                if self.schema["attention_mask"]:
                    feeds[self.schema["attention_mask"]] = np.ones((1, len(generated_ids)), dtype=np.int64)
                
                # Add empty KV cache for first step
                if self.schema["has_kv"]:
                    for name in self.schema["kv_inputs"]:
                        meta = self.schema["input_meta"][name]
                        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
                        feeds[name] = np.zeros(shape, dtype=np.float32)
                
                outs = self.session.run(None, feeds)
                logits = outs[0]
                if self.schema["has_kv"]:
                    past = outs[1:]
            else:
                # Subsequent steps: use single token + cache
                last_token = generated_ids[-1]
                logits, past = self._step_with_cache(last_token, past, seq_pos, len(generated_ids))
            
            # Get logits for last position
            last_logits = logits[0, -1, :]
            backup_logits = last_logits.copy()
            
            # Apply penalties
            last_logits = self._apply_penalties(last_logits, generated_ids, rep_penalty, 
                                              freq_lambda=freq_lambda, pres_lambda=pres_lambda)
            
            # Block repeating n-grams
            if ngram_block > 1:
                last_logits = self._block_repeating_ngrams(last_logits, generated_ids, ngram_block)
            
            # Apply EOS penalty
            last_logits = self._apply_eos_penalty(last_logits, step)
            
            # Apply function-word bias for first 20 tokens
            if step < 20:
                last_logits = self._apply_logit_bias(last_logits, self.function_word_tokens, bias=-0.9)
            
            # Check for short token repeats
            if self._detect_short_token_repeat(generated_ids):
                # Ban short tokens for 2 steps
                last_logits[list(self.short_tokens)] = -np.inf
            
            # Punctuation shaping every 15 tokens
            if step % 15 == 0 and step > 0:
                last_logits = self._apply_logit_bias(last_logits, self.punctuation_tokens, bias=+0.4)
            
            # Check for loops and adjust parameters
            current_temp = temperature
            current_top_p = top_p
            
            if step > 0:  # After first token
                loop_detected, frequent_token = loop_detector.detect_loop()
                if loop_detected:
                    if loop_detector.loop_count == 1:  # Print only once
                        print(f"🔄 Loop detected with token {frequent_token}, applying countermeasures...")
                    current_temp = min(temperature + 0.15, 1.25)  # Increase temperature
                    current_top_p = max(top_p - 0.05, 0.85)     # Tighten top-p
                    loop_detector.ban_token(frequent_token, duration=3)
                    # Apply strong logit bias to problematic token
                    last_logits = self._apply_logit_bias(last_logits, [frequent_token], bias=-2.5)
            
            # Apply banned tokens
            for banned_token in loop_detector.banned_tokens:
                last_logits[banned_token] = -np.inf
            
            # Sample next token
            next_id = self._optimized_sample(
                last_logits, k=top_k, p=current_top_p, temperature=current_temp, rng=rng,
                min_p=0.10, min_tokens_to_keep=4, backup_logits=backup_logits
            )
            
            # Update loop detector
            loop_detector.add_token(next_id)
            
            generated_ids.append(next_id)
            seq_pos += 1
            
            # Stop if EOS token
            if next_id == self.tokenizer.eos_token_id:
                break
        
        # Decode and return
        new_tokens = generated_ids[len(prompt_ids):]
        return self.decode(new_tokens)
    
    def _step_with_cache(self, token_id, past, seq_pos, attn_len):
        """Run single step with KV cache."""
        feeds = {self.schema["input_ids"]: np.array([[token_id]], dtype=np.int64)}
        if self.schema["attention_mask"]:
            feeds[self.schema["attention_mask"]] = np.ones((1, attn_len), dtype=np.int64)

        if self.schema["has_kv"]:
            if past is None:
                for name in self.schema["kv_inputs"]:
                    meta = self.schema["input_meta"][name]
                    shape = [d if isinstance(d, int) else 1 for d in meta.shape]
                    feeds[name] = np.zeros(shape, dtype=np.float32)
            else:
                for name, arr in zip(self.schema["kv_inputs"], past):
                    feeds[name] = arr

        outs = self.session.run(None, feeds)
        logits = outs[0]
        kv_out = outs[1:] if self.schema["has_kv"] else None
        return logits, kv_out
    
    def _apply_penalties(self, logits, generated_ids, rep_penalty=1.24, last_n=128,
                        freq_lambda=0.62, pres_lambda=0.22):
        """Apply repetition and frequency penalties."""
        if not generated_ids:
            return logits
        out = logits.astype(np.float32, copy=True)
        window = generated_ids[-last_n:] if last_n > 0 else generated_ids
        uniq, counts = np.unique(window, return_counts=True)
        if rep_penalty and rep_penalty > 1.0:
            out[uniq] /= rep_penalty
        out[uniq] -= pres_lambda
        out[uniq] -= freq_lambda * counts
        return out
    
    def _block_repeating_ngrams(self, logits, generated_ids, n=4):
        """Block repeating n-grams."""
        if n <= 1 or len(generated_ids) < n-1:
            return logits
        bans = {}
        for i in range(len(generated_ids) - (n - 1)):
            key = tuple(generated_ids[i:i+n-1])
            nxt = generated_ids[i+n-1]
            bans.setdefault(key, set()).add(nxt)
        prefix = tuple(generated_ids[-(n-1):])
        if prefix in bans:
            out = logits.copy()
            out[list(bans[prefix])] = -np.inf
            return out
        return logits
    
    def _apply_logit_bias(self, logits, token_ids, bias=-1.0):
        """Apply bias to specific tokens."""
        out = logits.copy()
        for token_id in token_ids:
            out[token_id] += bias
        return out
    
    def _apply_eos_penalty(self, logits, step, eos_token_id=50256, ban_steps=60):
        """Ban EOS for first N steps, then apply soft penalty."""
        out = logits.copy()
        if step < ban_steps:
            out[eos_token_id] = -np.inf  # Hard ban
        else:
            out[eos_token_id] -= 1.0  # Soft penalty
        return out
    
    def _detect_short_token_repeat(self, generated_ids, max_short=3, window=6):
        """Detect if too many short tokens in recent window."""
        if len(generated_ids) < window:
            return False
        
        recent_tokens = generated_ids[-window:]
        short_count = sum(1 for token_id in recent_tokens if token_id in self.short_tokens)
        
        return short_count >= max_short
    
    def _softmax(self, x):
        """Stable softmax implementation."""
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=-1e10, posinf=1e10, neginf=-1e10)
        
        if np.all(x == x[0]):
            return np.ones_like(x) / len(x)
        
        x_max = np.max(x)
        x_shifted = x - x_max
        e_x = np.exp(x_shifted)
        e_x = np.nan_to_num(e_x, nan=0.0, posinf=1e10, neginf=0.0)
        
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0 or not np.isfinite(sum_e_x):
            return np.ones_like(x) / len(x)
        
        result = e_x / sum_e_x
        result = np.nan_to_num(result, nan=0.0)
        
        result_sum = np.sum(result)
        if result_sum == 0:
            return np.ones_like(x) / len(x)
        
        return result / result_sum
    
    def _top_k_filter(self, logits, k=0, min_tokens_to_keep=4):
        """Apply top-k filtering with minimum tokens guarantee."""
        if k and k < logits.shape[-1]:
            thresh = np.partition(logits, -k)[-k]
            logits[logits < thresh] = -np.inf
        
        # Ensure minimum tokens are kept
        finite_count = np.sum(np.isfinite(logits))
        if finite_count < min_tokens_to_keep:
            top_indices = np.argpartition(logits, -min_tokens_to_keep)[-min_tokens_to_keep:]
            logits = np.full_like(logits, -np.inf)
            logits[top_indices] = 0
        return logits
    
    def _top_p_filter(self, logits, p=1.0, min_p=0.10, min_tokens_to_keep=4):
        """Apply top-p filtering with minimum probability floor."""
        probs = self._softmax(logits.copy())
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        csum = np.cumsum(sorted_probs)
        keep = csum <= p
        
        # Apply min_p floor
        max_prob = np.max(probs)
        min_prob_threshold = min_p * max_prob
        min_p_keep = probs >= min_prob_threshold
        
        # Combine both conditions
        keep = keep | min_p_keep
        
        # Ensure minimum tokens are kept
        if np.sum(keep) < min_tokens_to_keep:
            keep = np.zeros_like(keep, dtype=bool)
            keep[order[:min_tokens_to_keep]] = True
        
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        logits[~mask] = -np.inf
        return logits
    
    def _optimized_sample(self, logits, k=70, p=0.95, temperature=1.10, rng=np.random,
                         min_p=0.10, min_tokens_to_keep=4, backup_logits=None):
        """Optimized sampling with comprehensive safety nets."""
        # Store backup for safety
        if backup_logits is None:
            backup_logits = logits.copy()
        
        # Handle NaN and inf values
        logits = np.asarray(logits, dtype=np.float32)
        logits = np.nan_to_num(logits, nan=-1e10, posinf=1e10, neginf=-1e10)
        
        # Apply temperature with floor
        temperature = max(temperature, 0.7)
        l = logits / max(temperature, 1e-8)
        
        # Apply top-k filter
        if k and k > 0: 
            l = self._top_k_filter(l, k, min_tokens_to_keep)
        
        # Apply top-p filter
        if p < 1.0: 
            l = self._top_p_filter(l, p, min_p, min_tokens_to_keep)
        
        # Safety check: if no finite logits remain, restore backup
        if not np.any(np.isfinite(l)):
            l = backup_logits.copy()
            l = np.nan_to_num(l, nan=-1e10, posinf=1e10, neginf=-1e10)
            # Keep at least the top token
            top_idx = np.argmax(l)
            l = np.full_like(l, -np.inf)
            l[top_idx] = 0
        
        # Get probabilities
        probs = self._softmax(l)
        
        # Final safety check
        if np.any(np.isnan(probs)) or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        
        probs = np.nan_to_num(probs, nan=0.0)
        probs = probs / np.sum(probs)
        
        # Sample
        try:
            return int(rng.choice(len(probs), p=probs))
        except ValueError:
            return int(np.argmax(probs))


class EnhancedLoopDetector:
    """Enhanced loop detection with cooldown and banning."""
    
    def __init__(self, window_size=32, threshold=0.45):
        self.window_size = window_size
        self.threshold = threshold
        self.recent_tokens = deque(maxlen=window_size)
        self.banned_tokens = {}  # token_id -> steps_remaining
        self.cooldown = 0
        self.loop_count = 0
        
    def add_token(self, token):
        self.recent_tokens.append(token)
        # Reduce ban duration
        self.banned_tokens = {k: v-1 for k, v in self.banned_tokens.items() if v > 1}
        # Reduce cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def detect_loop(self):
        if len(self.recent_tokens) < self.window_size or self.cooldown > 0:
            return False, None
        
        # Count token frequencies
        token_counts = defaultdict(int)
        for token in self.recent_tokens:
            token_counts[token] += 1
        
        if not token_counts:
            return False, None
            
        most_frequent_token = max(token_counts, key=token_counts.get)
        frequency = token_counts[most_frequent_token] / len(self.recent_tokens)
        
        return frequency >= self.threshold, most_frequent_token
    
    def ban_token(self, token, duration=3):
        self.banned_tokens[token] = duration
        self.cooldown = 2
        self.loop_count += 1


# ==========================================================
# Usage Examples and Presets
# ==========================================================

def create_sampler(model_path=None, use_cuda=False):
    """Create and return an ONNX GPT-2 sampler instance."""
    return ONNXGPT2Sampler(model_path=model_path, use_cuda=use_cuda)

# Recommended presets for different use cases
PRESETS = {
    "int8_optimized": {
        "temperature": 1.10,
        "top_k": 70,
        "top_p": 0.95,
        "rep_penalty": 1.24,
        "freq_lambda": 0.62,
        "pres_lambda": 0.22,
        "ngram_block": 4
    },
    "fp32_standard": {
        "temperature": 1.05,
        "top_k": 80,
        "top_p": 0.94,
        "rep_penalty": 1.20,
        "freq_lambda": 0.60,
        "pres_lambda": 0.20,
        "ngram_block": 3
    },
    "creative": {
        "temperature": 1.15,
        "top_k": 100,
        "top_p": 0.92,
        "rep_penalty": 1.15,
        "freq_lambda": 0.50,
        "pres_lambda": 0.15,
        "ngram_block": 3
    },
    "conservative": {
        "temperature": 0.95,
        "top_k": 50,
        "top_p": 0.98,
        "rep_penalty": 1.30,
        "freq_lambda": 0.70,
        "pres_lambda": 0.30,
        "ngram_block": 4
    }
}

# Example usage
if __name__ == "__main__":
    # Create sampler
    sampler = create_sampler()
    
    # Use optimized preset for INT8
    preset = PRESETS["int8_optimized"]
    
    # Generate text
    prompt = "Coastal mornings start cool under a low gray deck. By noon, sea breeze clears the haze."
    output = sampler.generate(prompt, max_new_tokens=64, **preset)
    
    print("Generated text:")
    print("-----\n" + output)
