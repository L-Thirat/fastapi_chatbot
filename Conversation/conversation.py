from pysentimiento import create_analyzer
import collections
import re
from typing import Tuple, Any
from emoji import demojize

emotion_mapping = {
    'joy': "happy",
    'anger': "angry"
}


# msg constructor and formatter
class character_msg_constructor:
    def __init__(self, name, char_persona):
        self.name = name
        self.persona = char_persona.replace("\n", " ")
        self.emotion_analyzer = create_analyzer(task="emotion", lang="en")
        # self.split_counter = 0
        self.history_loop_cache = []

    def construct_msg(self, text: str):
        if self.history_loop_cache:
            if len(self.history_loop_cache) > 6:  # limit conversation history to prevent memory leak
                self.history_loop_cache = self.history_loop_cache[:2]
        else:
            # conversation_template = f"""{self.name}'s Persona: {self.persona}\n"""
            # conversation_template = f"""<|user|>\n{self.persona}</s>\n<|assistant|>\nได้ค่ะ โอนี่จัง\n"""
            self.history_loop_cache = [
                # {"role": "system", "content": template},
                {"role": "user", "content": f"""{self.persona}"""},
                {"role": "assistant"}#, "content": f"""オッケー、お兄ちゃん</s>"""}
            ]
        self.history_loop_cache.append({"role": "user", "content": text})

    # conversation formatter
    def get_current_converse(self, data: str, conversation_text: str) -> list:
        conversation_text = conversation_text.replace("<|assistant|>", "Luna: ")
        return [data , conversation_text]
        # splited = [x.strip() for x in conversation_text.split('\n') if x != '']
        # conversation_list = []
        # conversation_line_count = 0
        # for idx, thisline in enumerate(splited):
        #     holder = conversation_line_count
        #     if thisline.startswith(f'{self.name}:') or thisline.startswith('You:'):  # if found talking line
        #         holder += 1
        #
        #     if holder > conversation_line_count:  # append talking line at each found
        #         conversation_list.append(thisline)
        #         conversation_line_count = holder
        #
        #     elif conversation_line_count > 0:  # concat conversation into the line before if no new converse line found
        #         conversation_list[-1] = f'{conversation_list[-1].strip()} {thisline.strip()}'
        #
        # return conversation_list

    def emotion_analyze(self, text: str) -> tuple[str | Any, str]:
        emotion_to_express = 'netural'

        emotions_text = demojize(text)
        text = re.sub(r':\w+:', '', emotions_text)
        emotions_text = re.sub("[^A-Za-z]", " ", emotions_text).strip()
        if emotions_text:
            # emotions_text = re.findall(r'\((.*?)\)', emotions_text)
            emotions = self.emotion_analyzer.predict(emotions_text).probas
            ordered = dict(sorted(emotions.items(), key=lambda x: x[1]))
            ordered = [k for k, v in ordered.items()]  # top two emotion
            emotion_to_express = emotion_mapping.get(ordered[-1], emotion_to_express)
        # text = re.sub(r'\([^)]*\)', '', emotions_text)  # remove action/emotion inside (...)

        return emotion_to_express, text

    def clean_emotion_action_text_for_speech(self, text):
        clean_text = re.sub(r'\*.*?\*', '', text)  # remove *action* from text
        clean_text = clean_text.replace(f'{self.name}:', '')  # replace -> name: "dialog"
        return clean_text