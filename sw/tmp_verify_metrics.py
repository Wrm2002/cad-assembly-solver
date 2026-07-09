import csv

with open('C:/Users/11049/Desktop/Model_match/sw/data/functional_results/candidate_scores.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)
true_rows = [r for r in rows if r['evaluation_is_true_group'] == 'True']
selected_true = [r for r in true_rows if r['review_queue_state'] == 'selected']
deferred_true = [r for r in true_rows if r['review_queue_state'] == 'deferred']
rejected_true = [r for r in true_rows if r['final_decision'] == 'rejected']
accepted = [r for r in rows if r['final_decision'] == 'accepted']
review_selected = [r for r in rows if r['review_queue_state'] == 'selected']
pose_valid = [r for r in rows if r['pose_status'] == 'valid']
pose_failed = [r for r in rows if r['pose_status'] == 'failed']
pose_uncertain = [r for r in rows if r['pose_status'] == 'uncertain']

print(f'Total rows: {total}')
print(f'Accepted: {len(accepted)}')
print(f'True groups total: {len(true_rows)}')
print(f'True in review-selected (frontier): {len(selected_true)}')
print(f'True in review-deferred: {len(deferred_true)}')
print(f'True in rejected: {len(rejected_true)}')
print(f'Review selected total: {len(review_selected)}')
print(f'Pose valid: {len(pose_valid)}')
print(f'Pose failed: {len(pose_failed)}')
print(f'Pose uncertain/not checked: {len(pose_uncertain)}')
print()
print('True group details:')
for r in true_rows:
    print(f'  {r["candidate_id"]} parts={r["group_size"]} decision={r["final_decision"]} queue={r["review_queue_state"]} pose={r["pose_status"]} reasons={r["decision_reasons"][:80]}')
