# DEBUG
# python main.py \
#     --model_name_or_path t5-base \
#     --train_file data/axis_evals_train.csv \
#     --validation_file data/axis_evals_val.csv \
#     --test_file data/axis_evals_test.csv \
#     --max_length 512 \
#     --article_column article \
#     --summary_column summary_text \
#     --label_column label \
#     --output_dir reward_classification \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --max_train_samples 50 \
#     --max_eval_samples 10 \
#     --max_predict_samples 10 \
#     --batch_size 8 \
#     --learning_rate 3e-5 \
#     --num_epochs 3 \
#     --overwrite_output_dir


python main.py \
    --model_name_or_path t5-base \
    --train_file data/axis_evals_train.csv \
    --validation_file data/axis_evals_val.csv \
    --test_file data/axis_evals_test.csv \
    --max_length 512 \
    --article_column article \
    --summary_column summary_text \
    --label_column label \
    --output_dir reward_classification \
    --do_train \
    --do_eval \
    --do_predict \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 10 \
    --overwrite_output_dir