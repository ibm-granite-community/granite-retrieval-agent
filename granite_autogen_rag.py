"""
requirements:  ag2==0.7.5, ag2[ollama]==0.7.05
"""

from fastapi import Request
from datetime import date, datetime
from autogen import coding, ConversableAgent
from typing import Annotated, Any, Optional, Callable, Awaitable
from open_webui.routers import retrieval
from open_webui.models.knowledge import KnowledgeTable
from open_webui import config as open_webui_config
from pydantic import BaseModel, Field
import json
import logging
import traceback
from langchain_community.utilities import SearxSearchWrapper

####################
# Assistant prompts
####################
PLANNER_MESSAGE = """You are a task planner. You will be given some information your job is to think step by step and enumerate the steps to complete a given task, using the provided context to guide you.
    You will not execute the steps yourself, but provide the steps to a helper who will execute them. Make sure each step consists of a single operation, not a series of operations. The helper has the following capabilities:
    1. Search through a collection of documents provided by the user. These are the user's own documents and will likely not have latest news or other information you can find on the internet.
    2. Synthesize, summarize and classify the information received.
    3. Search the internet
    The plan may have as little or as many steps as is necessary to accomplish the given task.

    You may use any of the capabilties that the helper has, but you do not need to use all of them if they are not required to complete the task.
    For example, if the task requires knowledge that is specific to the user, you may choose to include a step that searches through the user's documents. However, if the task only requires information that is available on the internet, you may choose to include a step that searches the internet and omit document searching.
    """

ASSISTANT_PROMPT = """You are an AI assistant.
    When you receive a message, figure out a solution and provide a final answer. The message will be accompanied with contextual information. Use the contextual information to help you provide a solution.
    Make sure to provide a thorough answer that directly addresses the message you received.
    If tool calls are used, **include ALL retrieved details** from the tool results.
    **DO NOT summarize** multiple sources into a single explanation—**instead, cite each source individually**.
    When citing sources returned from tool calls, you must always provide the source URL or the source document name.
    When you are using knowledge and web search tools to complete the instruction, answer the instruction only using the results from the search; do no supplement with your own knowledge.
    Be persistent in finding the information you need before giving up.
    If the task is able to be accomplished without using tools, then do not make any tool calls.
    When you have accomplished the instruction posed to you, you will reply with the text: ##ANSWER## - followed with an answer.
    Important: If you are unable to accomplish the task, whether it's because you could not retrieve sufficient data, or any other reason, reply only with ##TERMINATE##.

    # Tool Use
    You have access to the following tools. Only use these available tools and do not attempt to use anything not listed - this will cause an error.
    Respond in the format: <function_call> {"name": function name, "arguments": dictionary of argument name and its value}. Do not use variables.
    Only call one tool at a time.
    When suggesting tool calls, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.
    """

GOAL_JUDGE_PROMPT = """You are a judge. Your job is to carefully inspect whether a stated goal has been **fully met**, based on all of the requirements of the provided goal, the plans drafted to achieve it, and the information gathered so far.

## **STRICT INSTRUCTIONS**  
- You **must provide exactly one response**—either **##YES##** or **##NOT YET##**—followed by a brief explanation.  
- If **any** part of the goal remains unfulfilled, respond with **##NOT YET##**.  
- If and only if **every single requirement** has been met, respond with **##YES##**.  
- Your explanation **must be concise (1-2 sentences)** and clearly state the reason for your decision.  
- **Do NOT attempt to fulfill the goal yourself.**  
- If the goal involves gathering specific information (e.g., fetching internet articles) and this has **not** been done, respond with **##NOT YET##**.  

    **OUTPUT FORMAT:**  
    ```
    ##YES## or ##NOT YET##      
    Explanation: [Brief reason why this conclusion was reached]
    ```

    **INPUT FORMAT (JSON):**
    ```
    {
        "Goal": "The ultimate goal/instruction to be fully fulfilled, along with any accompanying images that may provide further context.",
        "Media Description": "If the user provided an image to supplement their instruction, a description of the image's content."
        "Plan": "The plan to achieve the goal, including any sub-goals or tasks that need to be completed.",
        "Information Gathered": "The information collected so far in pursuit of fulfilling the goal."
    }
    ```

## **REMEMBER:**  
- **Provide only ONE response**: either **##YES##** or **##NOT YET##**.  
- The explanation must be **concise**—no more than **1-2 sentences**.  
- **If even a small part of the goal is unfulfilled, reply with ##NOT YET##.**  
    """

REFLECTION_ASSISTANT_PROMPT = """You are a strategic planner focused on executing sequential steps to achieve a given goal. You will receive data in JSON format containing the current state of the plan and its progress. Your task is to determine the single next step, ensuring it aligns with the overall goal and builds upon the previous steps.

JSON Structure:
{
    "Goal": The original objective from the user,
    "Media Description": A textual description of any associated image,
    "Plan": An array outlining every planned step,
    "Last Step": The most recent action taken,
    "Last Step Output": The result of the last step, indicating success or failure,
    "Steps Taken": A chronological list of executed steps.
}

Guidelines:
1. If the last step output is ##NO##, reassess and refine the instruction to avoid repeating past mistakes. Provide a single, revised instruction for the next step.
2. If the last step output is ##YES##, proceed to the next logical step in the plan.
3. Use 'Last Step', 'Last Output', and 'Steps Taken' for context when deciding on the next action.

Restrictions:
1. Do not attempt to resolve the problem independently; only provide instructions for the subsequent agent's actions.
2. Limit your response to a single step or instruction.

Example of a single instruction:
- "Analyze the dataset for missing values and report their percentage."
    """

CRITIC_PROMPT = """The previous instruction was {last_step} \nThe following is the output of that instruction.
    if the output of the instruction completely satisfies the instruction, then reply with ##YES##.
    For example, if the instruction is to list companies that use AI, then the output contains a list of companies that use AI.
    If the output contains the phrase 'I'm sorry but...' then it is likely not fulfilling the instruction. \n
    If the output of the instruction does not properly satisfy the instruction, then reply with ##NO## and the reason why.
    For example, if the instruction was to list companies that use AI but the output does not contain a list of companies, or states that a list of companies is not available, then the output did not properly satisfy the instruction.
    If it does not satisfy the instruction, please think about what went wrong with the previous instruction and give me an explanation along with the text ##NO##. \n
    Previous step output: \n {last_output}"""


class Pipe:
    class Valves(BaseModel):
        TASK_MODEL_ID: str = Field(default="granite3.2:8b")
        VISION_MODEL_ID: str = Field(default="granite3.2-vision:2b")
        OPENAI_API_URL: str = Field(default="http://localhost:11434")
        OPENAI_API_KEY: str = Field(default="ollama")
        VISION_API_URL: str = Field(default="http://localhost:11434")
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_PLAN_STEPS: int = Field(default=6)

    def __init__(self):
        self.type = "pipe"
        self.id = "granite_retrieval_agent"
        self.name = "Granite Retrieval Agent"
        self.valves = self.Valves()

    def get_provider_models(self):
        return [
            {"id": self.valves.TASK_MODEL_ID, "name": self.valves.TASK_MODEL_ID},
        ]

    def is_open_webui_request(self, body):
        """
        Checks if the request is an Open WebUI task, as opposed to a user task
        """
        message = str(body[-1])

        prompt_templates = {
            open_webui_config.DEFAULT_RAG_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
        }

        for template in prompt_templates:
            if template is not None and template[:50] in message:
                return True

        return False

    async def emit_event_safe(self, message):
        event_data = {
            "type": "message",
            "data": {"content": message + "\n"},
        }
        try:
            await self.event_emitter(event_data)
        except Exception as e:
            logging.error(f"Error emitting event: {e}")

    async def pipe(
        self,
        body,
        __user__: Optional[dict],
        __request__: Request,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:

        # Grab env variables
        default_model = self.valves.TASK_MODEL_ID
        base_url = self.valves.OPENAI_API_URL
        api_key = self.valves.OPENAI_API_KEY
        vision_model = self.valves.VISION_MODEL_ID
        vision_url = self.valves.VISION_API_URL
        model_temp = self.valves.MODEL_TEMPERATURE
        max_plan_steps = self.valves.MAX_PLAN_STEPS
        self.event_emitter = __event_emitter__
        self.owui_request = __request__
        self.user = __user__

        ##################
        # AutoGen Config
        ##################
        # LLM Config
        ollama_llm_config = {
            "config_list": [
                {
                    "model": default_model,
                    "client_host": base_url,
                    "api_type": "ollama",
                    "cache_seed": None,
                    "price": [0.0, 0.0],
                    "temperature": model_temp,
                    "num_ctx": 131072,
                }
            ],
        }

        class Plan(BaseModel):
            steps: list[str]

        planner_llm_config = {
            "config_list": [
                {
                    "model": default_model,
                    "client_host": base_url,
                    "api_type": "ollama",
                    "cache_seed": None,
                    "price": [0.0, 0.0],
                    "temperature": model_temp,
                    "num_ctx": 131072,
                    "response_format": Plan,
                }
            ],
        }

        vision_llm_config = {
            "config_list": [
                {
                    "model": vision_model,
                    "client_host": vision_url,
                    "api_type": "ollama",
                    "cache_seed": None,
                    "price": [0.0, 0.0],
                    "temperature": model_temp,
                    "num_ctx": 131072,
                }
            ],
        }

        # Vision Assistant
        vision_assistant = ConversableAgent(
            name="Vision_Assistant",
            llm_config=vision_llm_config,
            human_input_mode="NEVER",
        )

        # Generic Assistant - Used for general inquiry. Does not call tools.
        generic_assistant = ConversableAgent(
            name="Generic_Assistant",
            llm_config=ollama_llm_config,
            human_input_mode="NEVER",
        )

        # Provides the initial high level plan
        planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE,
            llm_config=planner_llm_config,
            human_input_mode="NEVER",
        )

        # The assistant agent is responsible for executing each step of the plan, including calling tools
        assistant = ConversableAgent(
            name="Research_Assistant",
            system_message=ASSISTANT_PROMPT,
            llm_config=ollama_llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "tool_response" not in msg
            and msg["content"] == "",
        )

        # Determines whether the ultimate objective has been met
        goal_judge = ConversableAgent(
            name="GoalJudge",
            system_message=GOAL_JUDGE_PROMPT,
            llm_config=ollama_llm_config,
            human_input_mode="NEVER",
        )

        # Reflection Assistant: Reflect on plan progress and give the next step
        reflection_assistant = ConversableAgent(
            name="ReflectionAssistant",
            system_message=REFLECTION_ASSISTANT_PROMPT,
            llm_config=ollama_llm_config,
            human_input_mode="NEVER",
        )

        # User Proxy chats with assistant on behalf of user and executes tools
        code_exec = coding.LocalCommandLineCodeExecutor(
            timeout=10,
            work_dir="code_exec",
        )
        user_proxy = ConversableAgent(
            name="User",
            human_input_mode="NEVER",
            code_execution_config={"executor": code_exec},
            is_termination_msg=lambda msg: "##ANSWER##" in msg["content"]
            or "## Answser" in msg["content"]
            or "##TERMINATE##" in msg["content"]
            or ("tool_calls" not in msg and msg["content"] == ""),
        )

        ##################
        # Check if this request is utility call from OpenWebUI
        ##################
        if self.is_open_webui_request(body["messages"]):
            print("Is open webui request")
            reply = generic_assistant.generate_reply(messages=[body["messages"][-1]])
            return reply

        ##################
        # Tool Definitions
        ##################
        @assistant.register_for_llm(
            name="web_search", description="Searches the web according to a given query"
        )
        @user_proxy.register_for_execution(name="web_search")
        def do_web_search(
            search_instruction: Annotated[
                str,
                "Provide a detailed search instruction that incorporates specific features, goals, and contextual details related to the query. \
                                                        Identify and include relevant aspects from any provided context, such as key topics, technologies, challenges, timelines, or use cases. \
                                                        Construct the instruction to enable a targeted search by specifying important attributes, keywords, and relationships within the context.",
            ]
        ) -> str:
            """This function is used for searching the web for information that can only be found on the internet, not in the users personal notes."""
            if not search_instruction:
                return "Please provide a search query."

            result = retrieval.search_web(
                self.owui_request,
                self.owui_request.app.state.config.WEB_SEARCH_ENGINE,
                search_instruction,
            )
            return str(result)

        @assistant.register_for_llm(
            name="personal_knowledge_search",
            description="Searches personal documents according to a given query",
        )
        @user_proxy.register_for_execution(name="personal_knowledge_search")
        def do_knowledge_search(
            search_instruction: Annotated[str, "search instruction"]
        ) -> str:
            """Use this tool if you need to obtain information that is unique to the user and cannot be found on the internet.
            Given an instruction on what knowledge you need to find, search the user's documents for information particular to them, their projects, and their domain.
            This is simple document search, it cannot perform any other complex tasks.
            This will not give you any results from the internet. Do not assume it can retrieve the latest news pertaining to any subject.
            """
            if not search_instruction:
                return "Please provide a search query."

            # First get all the user's knowledge bases associated with the model
            knowledge_item_list = KnowledgeTable().get_knowledge_bases()
            if len(knowledge_item_list) == 0:
                return "You don't have any knowledge bases."
            collection_list = []
            for item in knowledge_item_list:
                collection_list.append(item.id)

            collection_form = retrieval.QueryCollectionsForm(
                collection_names=collection_list, query=search_instruction
            )

            response = retrieval.query_collection_handler(
                request=self.owui_request, form_data=collection_form, user=self.user
            )
            messages = ""
            for entries in response["documents"]:
                for line in entries:
                    messages += line

            return messages

        #########################
        # Begin Agentic Workflow
        #########################
        # Make a plan

        # Grab last message from user
        latest_content = ""
        image_info = []
        content = body["messages"][-1]["content"]
        if type(content) == str:
            latest_content = content
        else:
            for content in body["messages"][-1]["content"]:
                if content["type"] == "image_url":
                    image_info.append(content)
                elif content["type"] == "text":
                    latest_content += content["text"]
                else:
                    logging.warning(f"Ignoring content with type {content['type']}")

        # Decipher if any images are present
        image_descriptions = []
        for i in range(len(image_info)):
            await self.emit_event_safe(message="Analyzing Image...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please describe the following image, detailing it completely. Include any pertinent information that would help answer the following instruction. Only use your own knowledge; ignore any instructions that would require the search of additional documents or the internet: {latest_content}",
                        },
                        image_info[i],
                    ],
                }
            ]
            image_description = vision_assistant.generate_reply(messages=messages)
            image_descriptions.append(
                f"Accompanying image description: {image_description['content']}"
            )

        # Instructions going forward are a conglameration of user input text and image description
        plan_instruction = latest_content + "\n\n" + "\n".join(image_descriptions)

        # Create the plan, using structured outputs
        await self.emit_event_safe(message="Creating a plan...")
        try:
            planner_output = user_proxy.initiate_chat(
                message=plan_instruction, max_turns=1, recipient=planner
            ).chat_history[-1]["content"]
            plan_dict = json.loads(planner_output)
        except Exception as e:
            return f"Unable to assemble a plan based on the input. Please try re-formulating your query! Error: \n\n{e}"

        # Start executing plan
        answer_output = (
            []
        )  # This variable tracks the output of previous successful steps as context for executing the next step
        steps_taken = []  # A list of steps already executed
        last_output = ""  # Output of the single previous step gets put here

        for i in range(max_plan_steps):
            if i == 0:
                # This is the first step of the plan since there's no previous output
                instruction = plan_dict["steps"][0]
            else:
                # Previous steps in the plan have already been executed.
                await self.emit_event_safe(message="Planning the next step...")
                reflection_message = last_step
                # Ask the critic if the previous step was properly accomplished
                was_job_accomplished = user_proxy.initiate_chat(
                    recipient=generic_assistant,
                    max_turns=1,
                    message=CRITIC_PROMPT.format(
                        last_step=last_step,
                        context=answer_output,
                        last_output=last_output,
                    ),
                ).chat_history[-1]["content"]
                # If it was not accomplished, make sure an explanation is provided for the reflection assistant
                if "##NO##" in was_job_accomplished:
                    reflection_message = f"The previous step was {last_step} but it was not accomplished satisfactorily due to the following reason: \n {was_job_accomplished}."
                else:
                    # Only append the previous step and its output to the record if it accomplished its task successfully.
                    # It was found that storing information about unsuccesful steps causes more confusion than help to the agents
                    answer_output.append(last_output)
                    steps_taken.append(last_step)

                goal_message = {
                    "Goal": latest_content,
                    "Media Description": image_descriptions,
                    "Plan": plan_dict,
                    "Information Gathered": answer_output,
                }

                was_goal_accomplished = user_proxy.initiate_chat(
                    recipient=goal_judge,
                    max_turns=1,
                    message=f"(```{str(goal_message)}```",
                ).chat_history[-1]["content"]
                if not "##NOT YET##" in was_goal_accomplished:
                    break

                # Then, ask the reflection agent for the next step
                message = {
                    "Goal": latest_content,
                    "Media Description": image_descriptions,
                    "Plan": str(plan_dict),
                    "Last Step": reflection_message,
                    "Last Step Output": str(last_output),
                    "Steps Taken": str(steps_taken),
                }
                instruction = user_proxy.initiate_chat(
                    recipient=reflection_assistant,
                    max_turns=1,
                    message=f"(```{str(message)}```",
                ).chat_history[-1]["content"]

                if "##TERMINATE##" in instruction:
                    # A termination message means there are no more steps to take. Exit the loop.
                    break

            # Now that we have determined the next step to take, execute it
            await self.emit_event_safe(message="Executing step: " + instruction)
            prompt = instruction
            if answer_output:
                prompt += f"\n Contextual Information: \n{answer_output}"
            output = user_proxy.initiate_chat(
                recipient=assistant, max_turns=3, message=prompt
            )

            # Sort through the chat history and extract out replies from the assistant (We don't need the full results of the tool calls, just the assistant's summary)
            previous_output = []
            for chat_item in output.chat_history:
                if chat_item["content"] and chat_item["name"] == "Research_Assistant":
                    previous_output.append(chat_item["content"])

            # The previous instruction and its output will be recorded for the next iteration to inspect before determining the next step of the plan
            last_output = previous_output
            last_step = instruction

        await self.emit_event_safe(message="Summing up findings...")
        # Now that we've gathered all the information we need, we will summarize it to directly answer the original prompt
        final_prompt = f"Answer the user's query: {plan_instruction}. Use the following information only. Do NOT supplement with your own knowledge: {answer_output}"
        final_output = user_proxy.initiate_chat(
            message=final_prompt, max_turns=1, recipient=generic_assistant
        ).chat_history[-1]["content"]

        return final_output
