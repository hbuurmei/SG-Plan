from transformers import pipeline


def query_model(question, model_name='mistralai/Mistral-7B-v0.1'):
    # Initialize the model and tokenizer
    generator = pipeline('text-generation', model=model_name)

    # Generate the response
    response = generator(question, truncation=True, max_length=50, num_return_sequences=1)

    # Extract and return the text of the first (and only) sequence
    return response[0]['generated_text']


objects = ["fan", "papers", "keyboard", "laptop", "mouse", "trashcan", "chair", "plant", "painting", "couch"]
question = "I want to sit comfortably, where shall I go?"
options_prompt = "Please choose from the following options: " + ", ".join(objects) + ". Please only type the name of the object."
prompt = question + " " + options_prompt
print("Prompt: ", prompt)
# response = query_model(prompt, 'mistralai/Mistral-7B-v0.1')
# print("Response: ", response)
