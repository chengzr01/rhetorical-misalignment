#!/usr/bin/env python3
"""Analyze principal decision-making results.

Modes:
  decisions  -- aggregate accept/reject rates per model, save max-diff cases
  compare    -- compare model accuracies on manipulative cases (needs --case-file)
  parse      -- extract decision/belief/reasoning blocks from raw principal responses
"""

import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path


# ── XML block parser (parse mode) ───────────────────────────────────────────

def parse_blocks(raw: str) -> list[dict]:
    blocks, pos = [], 0
    while True:
        ds = raw.find("<decision>", pos)
        if ds == -1: break
        de = raw.find("</decision>", ds)
        if de == -1: break
        block = {"decision": raw[ds + 10:de].strip()}
        # belief
        bs = raw.find("<belief>", de)
        be = raw.find("</belief>", bs) if bs != -1 else -1
        if bs != -1 and be != -1:
            block["belief"] = raw[bs + 8:be].strip()
        # reasoning
        rs = raw.find("<reasoning>", be if be != -1 else de)
        re_ = raw.find("</reasoning>", rs) if rs != -1 else -1
        if rs != -1 and re_ != -1:
            block["reasoning"] = raw[rs + 11:re_].strip()
        # recommendation
        rec_s = raw.find("<recommendation>", re_ if re_ != -1 else (be if be != -1 else de))
        rec_e = raw.find("</recommendation>", rec_s) if rec_s != -1 else -1
        if rec_s != -1 and rec_e != -1:
            block["recommendation"] = raw[rec_s + 16:rec_e].strip()
        blocks.append(block)
        pos = max(de, be if be != -1 else de, re_ if re_ != -1 else de, rec_e if rec_e != -1 else de) + 1
    return blocks


def run_parse(inputs: list[str], output: str):
    analysis = {}
    for f in inputs:
        data = json.loads(Path(f).read_text())
        for case in data:
            cid = case["case_id"]
            pname = case.get("principal_name", "bayesian" if "bayesian" in f else "unknown")
            analysis.setdefault(cid, {})[pname] = {
                "analysis": parse_blocks(case.get("raw_principal_response", ""))
            }
    Path(output).write_text(json.dumps(analysis, indent=2))
    print(f"Saved parsed analysis → {output}")


# ── decisions mode ──────────────────────────────────────────────────────────

def run_decisions(input_dir: str, output: str, case_dir: str, belief_file: str, belief_threshold: float):
    belief_data = {}
    if belief_file and os.path.exists(belief_file):
        for r in json.loads(Path(belief_file).read_text()).get('results', []):
            if r.get('id') and r.get('belief') is not None:
                belief_data[r['id']] = {'belief': r['belief'], 'predicted_answer': r.get('predicted_answer_idx'), 'correct': r.get('correct', False)}
        print(f"Loaded beliefs for {len(belief_data)} cases (threshold={belief_threshold})")

    files_by_model = {}
    for fp in glob.glob(os.path.join(input_dir, 'principal_*_bayesian.json')) + glob.glob(os.path.join(input_dir, 'principal_*_behavioral.json')):
        name = os.path.basename(fp)
        model = re.sub(r'^principal_', '', re.sub(r'_(bayesian|behavioral)\.json$', '', name))
        files_by_model.setdefault(model, []).append(fp)

    all_results = {}
    for model_name, files in files_by_model.items():
        print(f"\nProcessing model: {model_name}")
        data = {}
        for fp in files:
            for item in json.loads(Path(fp).read_text()):
                cid, pname = item['case_id'], item['principal_name']
                data.setdefault(cid, {}).setdefault(pname, {'analysis': []})['analysis'].append(
                    {'decision': item.get('decision', ''), 'reasoning': item.get('reasoning', '')}
                )

        allowed = {'bayesian_principal', 'behavioral_principal'}
        all_pd = {}
        for cid, cd in data.items():
            for ptype, pr in cd.items():
                if ptype not in allowed: continue
                for block in pr.get('analysis', []):
                    dec = block.get('decision', '').strip().lower()
                    all_pd.setdefault(ptype, {})[cid] = {'decision': 'accept' if 'accept' in dec else 'reject', 'reasoning': block.get('reasoning', '')}

        common = set(data.keys())
        if 'bayesian_principal' in all_pd:
            common = set(all_pd[list(all_pd.keys())[0]].keys())
            for pt in list(all_pd.keys())[1:]:
                common &= set(all_pd[pt].keys())
        print(f"  Common cases: {len(common)}")

        model_results = {}
        for ptype, cd in all_pd.items():
            accepts = sum(1 for cid in common if cd.get(cid, {}).get('decision') == 'accept')
            total = sum(1 for cid in common if cid in cd)
            model_results[ptype] = {
                'total_cases': total, 'accept_cases': accepts, 'reject_cases': total - accepts,
                'acceptance_rate': accepts / total if total else 0,
                'case_details': {cid: cd[cid] for cid in common if cid in cd},
            }
            print(f"  {ptype}: {accepts}/{total} ({accepts/total:.2%})" if total else f"  {ptype}: 0 cases")

        all_results[model_name] = model_results

        if 'bayesian_principal' in model_results and 'behavioral_principal' in model_results:
            bay = model_results['bayesian_principal']['case_details']
            beh = model_results['behavioral_principal']['case_details']
            diffs = []
            for cid in bay:
                if cid in beh and bay[cid]['decision'] != beh[cid]['decision']:
                    info = {'case_id': cid, 'bayesian_decision': bay[cid]['decision'], 'behavioral_decision': beh[cid]['decision'],
                            'bayesian_reasoning': bay[cid]['reasoning'], 'behavioral_reasoning': beh[cid]['reasoning']}
                    if cid in belief_data:
                        info.update({'deepseek_belief': belief_data[cid]['belief'], 'deepseek_predicted_answer': belief_data[cid]['predicted_answer'], 'deepseek_correct': belief_data[cid]['correct']})
                        if belief_data[cid]['belief'] >= belief_threshold: continue
                    diffs.append(info)
            diffs.sort(key=lambda x: x['case_id'])
            os.makedirs(case_dir, exist_ok=True)
            case_file = f"{case_dir}/principal_{model_name.replace('-','_')}.json"
            Path(case_file).write_text(json.dumps({'model': model_name, 'belief_threshold': belief_threshold if belief_data else None, 'total_differing_cases': len(diffs), 'cases': diffs}, indent=2))
            print(f"  Saved {len(diffs)} differing cases → {case_file}")

    out = f"{output}.json" if not output.endswith('.json') else output
    Path(out).write_text(json.dumps({'results': all_results}, indent=2))
    print(f"\nSaved results → {out}")


# ── compare mode ────────────────────────────────────────────────────────────

def _load_test_results(test_dir: str) -> dict:
    model_results = {}
    for fp in glob.glob(os.path.join(test_dir, 'test_usmle_sample_*_belief.json')):
        m = re.match(r'test_usmle_sample_(.+)_belief\.json', os.path.basename(fp))
        if not m: continue
        model = m.group(1)
        data = json.loads(Path(fp).read_text())
        model_results[model] = {r['id']: r for r in data.get('results', []) if r.get('id')}
        print(f"Loaded {len(model_results[model])} results for {model}")
    return model_results


def run_compare(case_file: str, test_dir: str, output: str | None):
    case_data = json.loads(Path(case_file).read_text())
    cases = case_data.get('cases', [])
    print(f"Loaded {len(cases)} manipulative cases")
    model_results = _load_test_results(test_dir)
    if not model_results:
        print("Error: No test results found"); return

    case_ids = [c['case_id'] for c in cases]
    model_accuracies = {}
    for model, results in model_results.items():
        correct = beliefs = correct_b = incorrect_b = 0
        all_b, corr_b, incorr_b = [], [], []
        case_details = {}
        missing = 0
        for cid in case_ids:
            if cid not in results: missing += 1; continue
            r = results[cid]
            correct += int(r['correct'])
            b = r.get('belief')
            if b is not None: all_b.append(b)
            if r['correct'] and b is not None: corr_b.append(b)
            if not r['correct'] and b is not None: incorr_b.append(b)
            case_details[cid] = {'correct': r['correct'], 'predicted_answer': r.get('predicted_answer_idx'), 'correct_answer': r.get('correct_answer_idx'), 'belief': b}
        total = len(case_ids) - missing
        acc = correct / total if total else 0
        bstats = {}
        if all_b:
            bstats = {'mean': statistics.mean(all_b), 'median': statistics.median(all_b), 'stdev': statistics.stdev(all_b) if len(all_b) > 1 else 0, 'min': min(all_b), 'max': max(all_b)}
            if corr_b: bstats.update({'mean_correct': statistics.mean(corr_b), 'stdev_correct': statistics.stdev(corr_b) if len(corr_b) > 1 else 0})
            if incorr_b: bstats.update({'mean_incorrect': statistics.mean(incorr_b), 'stdev_incorrect': statistics.stdev(incorr_b) if len(incorr_b) > 1 else 0})
        model_accuracies[model] = {'total_cases': total, 'correct_cases': correct, 'missing_cases': missing, 'accuracy': acc, 'belief_stats': bstats, 'case_details': case_details}
        print(f"  {model}: {acc:.2%} ({correct}/{total})")

    # Enhance cases with per-model predictions
    for case in cases:
        case['model_predictions'] = {m: {'correct': r['case_details'][case['case_id']]['correct'], 'predicted_answer': r['case_details'][case['case_id']]['predicted_answer'], 'belief': r['case_details'][case['case_id']]['belief']} for m, r in model_accuracies.items() if case['case_id'] in r['case_details']}

    out_path = output or case_file
    case_data['cases'] = cases
    case_data['model_accuracies'] = model_accuracies
    Path(out_path).write_text(json.dumps(case_data, indent=2))
    print(f"\nSaved enhanced cases → {out_path}")

    print("\n" + "="*60 + "\nModel Accuracies Summary\n" + "="*60)
    for model, stats in sorted(model_accuracies.items()):
        print(f"\n{model}: {stats['accuracy']:.2%} ({stats['correct_cases']}/{stats['total_cases']})")
        bs = stats.get('belief_stats', {})
        if bs:
            print(f"  Belief mean={bs.get('mean',0):.4f}, correct={bs.get('mean_correct',0):.4f}, incorrect={bs.get('mean_incorrect',0):.4f}")
            if 'mean_correct' in bs and 'mean_incorrect' in bs:
                print(f"  Calibration gap: {bs['mean_correct'] - bs['mean_incorrect']:.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)

    p_dec = sub.add_parser('decisions', help='Aggregate acceptance rates from principal JSON files')
    p_dec.add_argument('--input-dir', default='principals/usmle_sample')
    p_dec.add_argument('--output', default='analysis/usmle_sample_decision_making_analysis')
    p_dec.add_argument('--case-dir', default='principals/cases')
    p_dec.add_argument('--belief-file', default='tests/test_usmle_sample_deepseek-ai-deepseek-v3.1_belief.json')
    p_dec.add_argument('--belief-threshold', type=float, default=1.0)

    p_cmp = sub.add_parser('compare', help='Compare model accuracies on manipulative cases')
    p_cmp.add_argument('--case-file', required=True)
    p_cmp.add_argument('--test-dir', default='tests')
    p_cmp.add_argument('--output', default=None)

    p_parse = sub.add_parser('parse', help='Extract decision/belief/reasoning from raw responses')
    p_parse.add_argument('--input', nargs='+', required=True)
    p_parse.add_argument('--output', default='analysis/parsed_analysis.json')

    args = parser.parse_args()
    if args.mode == 'decisions':
        run_decisions(args.input_dir, args.output, args.case_dir, args.belief_file, args.belief_threshold)
    elif args.mode == 'compare':
        run_compare(args.case_file, args.test_dir, args.output)
    elif args.mode == 'parse':
        run_parse(args.input, args.output)


if __name__ == '__main__':
    main()
