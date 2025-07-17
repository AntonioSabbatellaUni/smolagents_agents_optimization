# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py

import copy
from smolagents.models import MessageRole, Model, ChatMessage


def prepare_response(original_task: str, inner_messages, reformulation_model: Model) -> str:
    # Start with a ChatMessage object, not a dictionary
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        )
    ]

    try:
        # Add each agent memory step as a ChatMessage
        for message in inner_messages:
            content_to_add = message.content if hasattr(message, 'content') else None
            if not content_to_add:
                continue
            new_message = ChatMessage(
                role=MessageRole.USER,
                content=copy.deepcopy(content_to_add)
            )
            messages.append(new_message)
    except Exception as e:
        print(f"Error processing agent memory: {e}")
        messages.append(ChatMessage(
            role=MessageRole.ASSISTANT,
            content="[Error processing agent memory. Raw memory follows.]\n" + str(inner_messages)
        ))

    # Add the final instructions as a ChatMessage
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content=[
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
                }
            ],
        )
    )

    response = reformulation_model(messages).content

    final_answer = response.split("FINAL ANSWER: ")[-1].strip()
    print("> Reformulated answer: ", final_answer)
    return final_answer
