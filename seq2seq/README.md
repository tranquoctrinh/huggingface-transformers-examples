# Training model sequence to sequence
I fine-tuning the T5-base model on the [CNN DailyMail](https://huggingface.co/datasets/cnn_dailymail) dataset.

To run script
```bash
python run_cnn_dailymail.py --model_name_or_path t5-base --do_train --do_eval --do_predict
```