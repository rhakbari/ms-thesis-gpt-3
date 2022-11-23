import openai

ft_discriminator = "davinci:ft-personal-2022-11-22-20-19-27"
ft_qa = "davinci:ft-personal-2022-11-22-21-04-53"

def apply_ft_discriminator(context, question, discriminator_model):
    """
    Apply the fine tuned discriminator to a question, to assess whether it can be answered from the context.
    """
    prompt = f"{context}\nQuestion: {question}\n Related:"
    result = openai.Completion.create(model=discriminator_model, prompt=prompt, max_tokens=1, temperature=0, top_p=1, n=1, logprobs=2)
    return result['choices'][0]['logprobs']['top_logprobs']

result = apply_ft_discriminator('Construction is the process of constructing a building or infrastructure. Construction differs from manufacturing in that manufacturing typically involves mass production of similar items without a designated purchaser, while construction typically takes place on location for a known client. Construction as an industry comprises six to nine percent of the gross domestic product of developed countries. Construction starts with planning,[citation needed] design, and financing and continues until the project is built and ready for use.', 
                        'What typically involves mass production of similar items without a designated purchaser?', ft_discriminator)
print(result)


def apply_ft_qa_answer(context, question, answering_model):
    """
    Apply the fine tuned discriminator to a question
    """
    prompt = f"{context}\nQuestion: {question}\nAnswer:"
    result = openai.Completion.create(model=answering_model, prompt=prompt, max_tokens=30, temperature=0, top_p=1, n=1, stop=['.','\n'])
    return result['choices'][0]['text']


answer = apply_ft_qa_answer('One of the most famous people born in Warsaw was Maria Skłodowska-Curie, who achieved international recognition for her research on radioactivity and was the first female recipient of the Nobel Prize. Famous musicians include Władysław Szpilman and Frédéric Chopin. Though Chopin was born in the village of Żelazowa Wola, about 60 km (37 mi) from Warsaw, he moved to the city with his family when he was seven months old. Casimir Pulaski, a Polish general and hero of the American Revolutionary War, was born here in 1745.', 
                        'How old was Chopin when he moved to Warsaw with his family?', ft_qa)

print(answer)