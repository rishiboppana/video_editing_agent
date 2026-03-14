from openai import OpenAI

client = OpenAI()

def run_agent(user_prompt):

    response = client.responses.create(
        model="gpt-4.1",
        input=user_prompt,
        tools=tools
    )

    tool_call = response.output[0]

    if tool_call.name == "trim_video":
        args = tool_call.arguments

        result = trim_video(
            args["video_path"],
            args["start_time"],
            args["end_time"]
        )

        return result