import asyncio
from typing import Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    VoxtralForConditionalGeneration,
)


class AIServices:
    def __init__(self, config=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._llm_initialized = False
        self.llm_model: Optional[AutoModelForCausalLM] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        self.llm_model_id: str = (
            getattr(config, "llm_model", "Qwen/Qwen3-30B-A3B-Instruct-2507")
            if config
            else "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )

        self._asr_initialized = False
        self.asr_model: Optional[VoxtralForConditionalGeneration] = None
        self.asr_processor: Optional[AutoProcessor] = None
        self.asr_model_id: str = (
            getattr(config, "asr_model", "mistralai/Voxtral-Mini-3B-2507")
            if config
            else "mistralai/Voxtral-Mini-3B-2507"
        )

    async def initialize_llm(self, model_name: Optional[str] = None) -> None:
        if self._llm_initialized:
            return

        model_name = model_name or self.llm_model_id
        logger.info(f"Initializing LLM model: {model_name}")
        try:
            loop = asyncio.get_event_loop()

            def _load_llm():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
                return tokenizer, model

            self.llm_tokenizer, self.llm_model = await loop.run_in_executor(None, _load_llm)
            self._llm_initialized = True
            logger.info(f"LLM model {self.llm_model_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise

    async def initialize_asr(self, model_name: Optional[str] = None) -> None:
        if self._asr_initialized:
            return

        model_name = model_name or self.asr_model_id
        logger.info(f"Initializing ASR model: {model_name}")
        try:
            loop = asyncio.get_event_loop()

            def _load_asr():
                processor = AutoProcessor.from_pretrained(model_name)
                model = VoxtralForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                return processor, model

            self.asr_processor, self.asr_model = await loop.run_in_executor(None, _load_asr)
            self._asr_initialized = True
            logger.info("ASR model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ASR model: {e}")
            raise

    async def generate_chat_response(
        self,
        message: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_new_tokens: int = 1024,
    ) -> str:
        if not self._llm_initialized or not self.llm_model or not self.llm_tokenizer:
            await self.initialize_llm()

        try:
            messages = [{"role": "user", "content": message}]
            loop = asyncio.get_event_loop()

            def _generate():
                assert self.llm_model is not None, "LLM model not initialized"
                assert self.llm_tokenizer is not None, "LLM tokenizer not initialized"

                text = self.llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(
                    self.llm_model.device
                )

                # Generate response
                generated_ids = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                )

                # Decode only the new tokens
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
                response = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True)
                return response.strip()

            response = await loop.run_in_executor(None, _generate)
            logger.info("Generated chat response successfully")
            return response

        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            return "Sorry, I encountered an error while processing your message."

    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: str = "en",
        max_new_tokens: int = 500,
    ) -> str:
        if not self._asr_initialized or not self.asr_model or not self.asr_processor:
            await self.initialize_asr()

        if not self.asr_model or not self.asr_processor:
            raise RuntimeError("ASR model not initialized")

        try:
            loop = asyncio.get_event_loop()

            def _transcribe():
                # Apply transcription request
                inputs = self.asr_processor.apply_transcription_request(
                    language=language,
                    audio=audio_file_path,
                    model_id="mistralai/Voxtral-Mini-3B-2507",
                )
                inputs = inputs.to(self.asr_model.device, dtype=torch.bfloat16)

                # Generate transcription
                with torch.no_grad():
                    outputs = self.asr_model.generate(**inputs, max_new_tokens=max_new_tokens)

                # Decode the output
                decoded_outputs = self.asr_processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                return decoded_outputs[0].strip() if decoded_outputs else ""

            transcription = await loop.run_in_executor(None, _transcribe)
            logger.info("Audio transcription completed successfully")
            return transcription

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return "Sorry, I couldn't transcribe the audio message."

    async def cleanup(self) -> None:
        """Clean up models to free memory."""
        logger.info("Cleaning up AI models")

        # Clear models
        if self.llm_model:
            del self.llm_model
            self.llm_model = None
        if self.llm_tokenizer:
            del self.llm_tokenizer
            self.llm_tokenizer = None
        if self.asr_model:
            del self.asr_model
            self.asr_model = None
        if self.asr_processor:
            del self.asr_processor
            self.asr_processor = None

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._llm_initialized = False
        self._asr_initialized = False
        logger.info("AI models cleanup completed")
