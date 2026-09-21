#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from sampling import sample_temperature, sample_top_k, sample_top_p, diversity_metrics
RESULTS = Path(__file__).resolve().parent / 'results'
rng = np.random.default_rng(42)
logits = np.log(1.0 / np.arange(1, 51) ** 1.1) + rng.normal(0, 0.05, size=50)
def main():
    rows=[]
    for name, fn in [('temp_0.5', lambda: sample_temperature(logits,0.5,rng)),('temp_1.5', lambda: sample_temperature(logits,1.5,rng)),('topk_5', lambda: sample_top_k(logits,5,1.0,rng)),('topp_0.9', lambda: sample_top_p(logits,0.9,1.0,rng))]:
        s=np.array([fn() for _ in range(1000)]); rows.append({'name':name, **diversity_metrics(s,50)})
    shot={'snapshot':'smoke_run','seed':42,'entropy_temp_0.5':rows[0]['entropy'],'entropy_temp_1.5':rows[1]['entropy'],'entropy_topp_0.9':rows[3]['entropy'],'unique_topk_5':rows[2]['unique_tokens'],'runtime_s':0.0}
    RESULTS.mkdir(exist_ok=True); (RESULTS/'JSON.shot').write_text(json.dumps(shot, indent=2)+'\n'); print(json.dumps(shot, indent=2))
if __name__ == '__main__':
    main()
