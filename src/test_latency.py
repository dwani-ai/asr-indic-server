import asyncio
import time
import requests
import torch
import torchaudio
from transformers import AutoModel
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# Load model (same as FastAPI)
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    revision="e9b71b369c048e2c6b634d4c131061c34e441179",
    trust_remote_code=True
)

# Test configuration
model_language = {
    "kannada": "kn", "hindi": "hi", "marathi": "mr",
    "tamil": "ta", "telugu": "te"
}

# Audio preprocessing function (same as FastAPI)
def preprocess_audio(audio_path):
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    return wav

async def benchmark_single_file(audio_path, lang_name, lang_code, decoder_type="ctc"):
    """Benchmark single audio file + decoder combo"""
    start_time = time.perf_counter()
    
    # Preprocess
    preprocess_start = time.perf_counter()
    wav = preprocess_audio(audio_path)
    preprocess_time = time.perf_counter() - preprocess_start
    
    # Inference
    inference_start = time.perf_counter()
    result = model(wav, lang_code, decoder_type)
    inference_time = time.perf_counter() - inference_start
    
    total_time = time.perf_counter() - start_time
    
    return {
        "language": lang_name,
        "decoder": decoder_type,
        "total_time_ms": total_time * 1000,
        "preprocess_time_ms": preprocess_time * 1000,
        "inference_time_ms": inference_time * 1000,
        "text": result,
        "audio_duration": wav.shape[1] / 16000  # seconds
    }

async def run_full_benchmark():
    """Run complete benchmark for all languages + both decoders"""
    
    # Update with your actual audio file paths
    test_files = {
        "kannada": "samples/kannada_sample_1.wav",
        "hindi": "samples/hindi_sample.wav", 
        "marathi": "samples/marathi_sample.wav",
        "tamil": "samples/tamil_sample.wav",
        "telugu": "samples/telugu_sample.wav"
    }
    
    results = []
    
    # Warmup run
    print("🔥 Warming up model...")
    warmup_wav = preprocess_audio(list(test_files.values())[0])
    model(warmup_wav, "hi", "ctc")
    
    print("\n⏱️  Running benchmarks...\n")
    
    # Test both CTC and RNNT for each language
    for lang_name, audio_path in test_files.items():
        lang_code = model_language[lang_name]
        
        print(f"Testing {lang_name.upper()}...")
        
        # CTC benchmark
        ctc_result = await benchmark_single_file(audio_path, lang_name, lang_code, "ctc")
        results.append(ctc_result)
        
        # RNNT benchmark  
        rnnt_result = await benchmark_single_file(audio_path, lang_name, lang_code, "rnnt")
        results.append(rnnt_result)
        
        print(f"  ✅ CTC: {ctc_result['total_time_ms']:.1f}ms")
        print(f"  ✅ RNNT: {rnnt_result['total_time_ms']:.1f}ms")
        print()
    
    return pd.DataFrame(results)

def print_results_table(results_df):
    """Pretty print benchmark results"""
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    # Summary table
    summary = results_df.groupby(['decoder', 'language']).agg({
        'total_time_ms': ['mean', 'std'],
        'inference_time_ms': 'mean',
        'audio_duration': 'first'
    }).round(1)
    
    print(summary)
    
    # Speed comparison
    ctc_avg = results_df[results_df['decoder'] == 'ctc']['total_time_ms'].mean()
    rnnt_avg = results_df[results_df['decoder'] == 'rnnt']['total_time_ms'].mean()
    
    print(f"\n⚡ SPEED COMPARISON")
    print(f"CTC Average:  {ctc_avg:.1f}ms ({ctc_avg/rnnt_avg:.1f}x faster)")
    print(f"RNNT Average: {rnnt_avg:.1f}ms")
    
    # Real-time factor (RTF)
    avg_duration = results_df['audio_duration'].mean()
    ctc_rtf = ctc_avg / (avg_duration * 1000)
    print(f"\n🎯 REAL-TIME FACTOR (lower = better)")
    print(f"CTC RTF:  {ctc_rtf:.4f}  (<< 1.0 = real-time capable)")

# Run the benchmark
async def main():
    results_df = await run_full_benchmark()
    print_results_table(results_df)
    
    # Save detailed results
    results_df.to_csv("asr_benchmark_results.csv", index=False)
    print(f"\n💾 Detailed results saved to 'asr_benchmark_results.csv'")

if __name__ == "__main__":
    asyncio.run(main())
