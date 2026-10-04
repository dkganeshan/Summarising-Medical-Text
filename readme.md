# **<u>Project 1: Summarising Medical Text</u>** 

The model was implemented using the rela�vely small FLAN-T5-small architecture so that fine-tuning could be performed on the available CPU-based local machine within prac�cal memory and training-�me constraints. Addi�onal resource-management techniques included a limited training batch size, gradient accumula�on and bounded input/output sequence lengths. 

## **Data Prepara�on and Training Model:** 

Text labelled FINDINGS and IMPRESSION were extracted from XML radiology reports. The extracted reports are stored in a pandas Data Frame and exported to “extracted_data.csv”. 

The data from “extracted_data.csv” is divided into training and tes�ng datasets. I assigned 85% of the records to training (train_data.csv) and 15% to tes�ng (test_data.csv), using a fixed random seed so the split is reproducible. 

The fine-tuned model has achieved F1 Score 0.65.  The model Evalua�on result and sample outs also the coding can be found in GitHub link below. <u>h�ps://github.com/dkganeshan/Summarising-Medical-Text</u> 

A further parameter-efficient approach can be considered for larger models is LoRA, which freezes the pretrained model weights and trains only a small number of addi�onal adapter parameters. Quan�sa�on or QLoRA could also be inves�gated to reduce memory requirements. 

For a full-scale implementa�on, training could be moved to GPU infrastructure using distributed mul�-GPU training. This would allow a larger model, larger effec�ve batch sizes, more extensive hyperparameter op�misa�on and poten�ally a larger and more diverse training dataset. 

Data-parallel training could distribute batches across mul�ple GPUs, while model-parallel or related distributed techniques could be considered if the model itself exceeds the memory of a single GPU. 

## **REST API documenta�on:** 

FastAPI REST API is used, and uvicorn is used to call the API. 

URL: <u>h�ps://my-ai-service-916269053394.europe-west1.run.app/</u> 

**Method:** POST 

Authen�ca�on required: No 

Data example: AP and lateral views were obtained. Bibasilar atelectasis and small le�-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular conges�on 

## **Success Response** 

Code: 200 

Content example: “Stable cardiomegaly with small le�-sided pleural effusion.” 

## **Error Response** 

Code: 400 

Content: “Error: Missing 'findings' field in request body” 

