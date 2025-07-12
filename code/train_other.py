import pandas as pd
from tqdm import tqdm
from collections import Counter
import logging
from utils_gpt4 import evaluate_on_training_data, generate_final_prediction, generate_final_prediction_context, generate_final_prediction_context_agent
import openai
# === Configuration ===
MODEL_NAME = "o1"
TEST_DATA = "test_cleaned.csv"
OUTPUT_FILE = "vf_of_code_to_focus_on/results/output_of_prediction/submission_gpt-o1.csv"
RAW_PRED_FILE = "vf_of_code_to_focus_on/results/output_of_prediction/submission_gpt-o1.csv"
NUM_PREDICTIONS = 5

#  openAPI key
openai.api_key="your-api-key-here"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("prediction_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# === Main ===
def main():
    try:
        df = pd.read_csv(TEST_DATA)
        logger.info(f"{len(df)} questions chargées depuis {TEST_DATA}")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des données : {e}")
        return

    all_final_predictions = []
    all_original_predictions = []
    total_stats = Counter()

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="🎯 Génération GPT"):
            final_pred, raw_preds = generate_final_prediction(row)
            all_final_predictions.append(final_pred)
            all_original_predictions.append(raw_preds)
            total_stats.update([p for p in raw_preds if p in ['a', 'b', 'c', 'd']])
    except KeyboardInterrupt:
        logger.warning("⏹️ Interruption par l'utilisateur.")

    submission = pd.DataFrame({'question_id': df['question_id']})
    for i in range(NUM_PREDICTIONS):
        submission[f'prediction_{i + 1}'] = [preds[i] for preds in all_final_predictions]
    submission.to_csv(OUTPUT_FILE, index=False)
    
    
    logger.info(f"📁 Fichier de soumission sauvegardé : {OUTPUT_FILE}")

    raw_df = pd.DataFrame(all_original_predictions, columns=[f'raw_pred_{i+1}' for i in range(NUM_PREDICTIONS)])
    raw_df.insert(0, 'question_id', df['question_id'])
    raw_df.to_csv(RAW_PRED_FILE, index=False)
    logger.info(f"📁 Fichier brut sauvegardé : {RAW_PRED_FILE}")

    total = sum(total_stats.values())
    logger.info("\n✅ Statistiques des prédictions valides :")
    for letter in ['a', 'b', 'c', 'd']:
        count = total_stats[letter]
        pct = (count / total) if total else 0
        logger.info(f"  {letter.upper()}: {count} fois ({pct:.2%})")

    is_valid = submission.drop('question_id', axis=1).isin(['a', 'b', 'c', 'd', 'invalid']).all().all()
    if not is_valid:
        logger.error("⚠️ Des prédictions non valides détectées.")
    else:
        logger.info("✔ Toutes les prédictions sont valides ou marquées 'invalid'.")

if __name__ == "__main__":
    main()
    evaluate_on_training_data()