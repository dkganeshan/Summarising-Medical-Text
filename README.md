Project 1: Summarising Medical Text

The model was implemented using the relatively small FLAN-T5-small architecture so that fine-tuning could be performed on the available CPU-based local machine within practical memory and training-time constraints. Additional resource-management techniques included a limited training batch size, gradient accumulation and bounded input/output sequence lengths.

Data Preparation and Training Model:
Text labelled FINDINGS and IMPRESSION were extracted from XML radiology reports. The extracted reports are stored in a pandas Data Frame and exported to “extracted_data.csv”.
The data from “extracted_data.csv” is divided into training and testing datasets. I assigned 85% of the records to training (train_data.csv) and 15% to testing (test_data.csv), using a fixed random seed so the split is reproducible.
The fine-tuned model has achieved F1 Score 0.65.  The model Evaluation result and sample outs also the coding can be found in GitHub link below.
https://github.com/dkganeshan/Summarising-Medical-Text

A further parameter-efficient approach can be considered for larger models is LoRA, which freezes the pre-trained model weights and trains only a small number of additional adapter parameters. Quantisation or QLoRA could also be investigated to reduce memory requirements.

For a full-scale implementation, training could be moved to GPU infrastructure using distributed multi-GPU training. This would allow a larger model, larger effective batch sizes, more extensive hyperparameter optimisation and potentially a larger and more diverse training dataset.   

Data-parallel training could distribute batches across multiple GPUs, while model-parallel or related distributed techniques could be considered if the model itself exceeds the memory of a single GPU.



REST API documentation:
FastAPI REST API is used, and uvicorn is used to call the API.
URL: https://my-ai-service-916269053394.europe-west1.run.app/
Method: POST
Authentication required: No
Data example: AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion
Success Response
Code: 200
Content example: “Stable cardiomegaly with small left-sided pleural effusion.”
Error Response
Code: 400
Content: “Error: Missing 'findings' field in request body” 
