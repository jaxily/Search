

python3 scripts/manage_data.py full-pull --years 15 --lookforward 12 --verbose


python3 main.py --data-file QQQ_20250809_155753_cleaned_v2.csv --ensemble-method Voting --skip-walkforward   

python3 main.py --data-file QQQ_20250809_155753_cleaned_v2.csv --ensemble-method Voting --skip-walkforward   



🎉 SUCCESS! Your QQQ Ensemble is Now Running!
📊 Results Summary
✅ What Just Happened:
9 models all fitted successfully with your QQQ data
3,769 samples processed (about 15 years of trading data)
29 features used (technical indicators + price/volume data)
Target: Binary classification (55.9% positive, 44.1% negative returns)


🚀 Performance Results:
Sharpe Ratio: 14.932 (excellent!)
Optimal Threshold: 0.500
Trade Count: 1,581 trades
Diversity Control: Applied to 3 highly correlated model pairs

📁 Artifacts Saved:
All results are saved in results/qqq_ensemble/ including:
OOF probabilities matrix
Optimal weights
Performance metrics
Ensemble model file
Version information

python3 scripts/run_ensemble.py --data_path QQQ_20250809_155753_cleaned_v2.csv --output_dir results/qqq_ensemble --transaction_cost 0.001 --slippage_bps 0.5 --optimization_method sharpe --n_splits 5
