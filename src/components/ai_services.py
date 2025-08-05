import asyncio
from typing import Dict, Optional, Sequence

import torch
from loguru import logger
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import (
    AudioChunk,
    RawAudio,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import LLM, SamplingParams


class AIServices:
    def __init__(self, config=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._llm_initialized = False
        self.llm_model: Optional[LLM] = None
        self.llm_model_id: str = (
            getattr(config, "llm_model", "Qwen/Qwen3-30B-A3B-Instruct-2507")
            if config
            else "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )

        self._asr_initialized = False
        self.asr_model: Optional[LLM] = None
        self.asr_tokenizer: Optional[MistralTokenizer] = None
        self.asr_model_id: str = (
            getattr(config, "asr_model", "mistralai/Voxtral-Mini-3B-2507")
            if config
            else "mistralai/Voxtral-Mini-3B-2507"
        )

    async def initialize_llm(self, model_name: Optional[str] = None) -> None:
        if self._llm_initialized:
            return

        model_name = model_name or self.llm_model_id
        logger.info(f"Initializing LLM model with vLLM: {model_name}")
        try:
            loop = asyncio.get_event_loop()

            def _load_llm():
                llm = LLM(
                    model=model_name,
                    dtype="auto",
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.5,
                    trust_remote_code=True,
                )
                return llm

            self.llm_model = await loop.run_in_executor(None, _load_llm)
            self._llm_initialized = True
            logger.info(f"LLM model {self.llm_model_id} initialized with vLLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise

    async def initialize_asr(self, model_name: Optional[str] = None) -> None:
        if self._asr_initialized:
            return

        model_name = model_name or self.asr_model_id
        logger.info(f"Initializing ASR model with vLLM: {model_name}")
        try:
            loop = asyncio.get_event_loop()

            def _load_asr():
                tokenizer = MistralTokenizer.from_hf_hub(model_name)
                limits = {"image": 0, "video": 0, "audio": 1}
                llm = LLM(
                    model=model_name,
                    tokenizer=model_name,
                    max_model_len=8192,
                    max_num_seqs=2,
                    limit_mm_per_prompt=limits,
                    config_format="mistral",
                    load_format="mistral",
                    tokenizer_mode="mistral",
                    enforce_eager=True,
                    enable_chunked_prefill=False,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.5,
                )
                return llm, tokenizer

            self.asr_model, self.asr_tokenizer = await loop.run_in_executor(None, _load_asr)
            self._asr_initialized = True
            logger.info("ASR model initialized successfully with vLLM")
        except Exception as e:
            logger.error(f"Failed to initialize ASR model: {e}")
            raise

    async def generate_chat_response(
        self,
        message: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 1024,
    ) -> str:
        if not self._llm_initialized or not self.llm_model:
            await self.initialize_llm()

        try:
            loop = asyncio.get_event_loop()

            def _generate():
                assert self.llm_model is not None, "LLM model not initialized"

                # Create sampling parameters with hard-coded values
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                )

                conversation: Sequence[Dict[str, str]] = [
                    {"role": "user", "content": message},
                ]

                # Generate response using vLLM
                outputs = self.llm_model.chat(conversation, sampling_params)  # type: ignore
                if outputs and outputs[0].outputs:
                    response = outputs[0].outputs[0].text.strip()
                    return response
                else:
                    return "Sorry, I couldn't generate a response."

            response = await loop.run_in_executor(None, _generate)
            logger.info("Generated chat response successfully with vLLM")
            return response

        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            return "Sorry, I encountered an error while processing your message."

    async def transcribe_audio(
        self,
        audio_file_path: str,
        max_tokens: int = 500,
    ) -> str:
        if not self._asr_initialized or not self.asr_model:
            await self.initialize_asr()

        try:
            loop = asyncio.get_event_loop()

            @logger.catch()
            def _transcribe():
                assert self.asr_model is not None, "ASR model not initialized"
                assert self.asr_tokenizer is not None, "ASR tokenizer not initialized"

                sampling_params = SamplingParams(
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                prompt = "Transcribe the audio."
                audio = Audio.from_file(audio_file_path, strict=False)

                text_chunk = TextChunk(text=prompt)
                audio_chunk = AudioChunk(input_audio=RawAudio.from_audio(audio))

                message = UserMessage(content=[audio_chunk, text_chunk])
                req = ChatCompletionRequest(messages=[message], model=self.asr_model_id)
                tokens = self.asr_tokenizer.encode_chat_completion(req)
                prompt_ids, audio = tokens.tokens, tokens.audios
                audio_and_sr = [(au.audio_array, au.sampling_rate) for au in audio]

                inputs = {
                    "multi_modal_data": {"audio": audio_and_sr},
                    "prompt_token_ids": prompt_ids,
                }
                outputs = self.asr_model.generate(
                    inputs,
                    sampling_params=sampling_params,
                )

                if outputs and outputs[0].outputs:
                    transcription = outputs[0].outputs[0].text.strip()
                    return transcription
                else:
                    return ""

            transcription = await loop.run_in_executor(None, _transcribe)
            logger.info(f"Audio transcribed with {self.asr_model_id}: {transcription}")
            return transcription

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
            return "Sorry, I couldn't transcribe the audio message."

    async def cleanup(self) -> None:
        logger.info("Cleaning up AI models")

        # Clear models
        if self.llm_model:
            self.llm_model = None
        if self.asr_model:
            self.asr_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._llm_initialized = False
        self._asr_initialized = False
        logger.info("AI models cleanup completed")
