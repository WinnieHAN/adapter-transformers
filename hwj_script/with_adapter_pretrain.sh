# pretrain
MAIN_PATH=../adapter-transformers/examples/my
cd ${MAIN_PATH}
GLUE_DIR=${MAIN_PATH}/glue_data
GLUE_TXT_DIR=${MAIN_PATH}/GLUE_data
# 
# WNLI RTE MNLI
for TASK_NAME in QQP
do
CUDA_VISIBLE_DEVICES=2 python run_mlm_adap_my.py  \
    --model_name_or_path bert-base-uncased \
    --line_by_line \
    --train_file ${GLUE_TXT_DIR}/$TASK_NAME/train.tsv.txt \
    --validation_file ${GLUE_TXT_DIR}/$TASK_NAME/dev.tsv.txt \
    --do_train \
    --do_eval \
    --save_total_limit 20 \
    --num_train_epochs 10 \
    --overwrite_output_dir \
    --output_dir temp_adp_mlm_${TASK_NAME}

done
