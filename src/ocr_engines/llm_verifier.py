"""
LLM-based OCR verification and correction.
Integrates GPT-4V, Qwen2-VL, or other multimodal LLMs for contextual validation.
"""

import logging
from typing import Dict, Optional, List
import re

# LLM imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available")


class LLMVerifier:
    """
    LLM-based verification for OCR results.
    
    Uses multimodal LLMs to:
    - Verify OCR accuracy
    - Correct errors using context
    - Validate format and range
    - Detect hallucinations
    """
    
    def __init__(self,
                 provider: str = 'openai',
                 model: str = 'gpt-4-vision-preview',
                 api_key: Optional[str] = None,
                 enable_verification: bool = True,
                 enable_correction: bool = True,
                 enable_validation: bool = True):
        """
        Initialize LLM verifier.
        
        Args:
            provider: 'openai', 'qwen', or 'local'
            model: Model identifier
            api_key: API key for commercial providers
            enable_verification: Enable LLM verification
            enable_correction: Enable LLM correction
            enable_validation: Enable rule-based validation
        """
        self.provider = provider
        self.model = model
        self.enable_verification = enable_verification
        self.enable_correction = enable_correction
        self.enable_validation = enable_validation
        
        # Initialize LLM client
        if provider == 'openai' and OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key)
            self.llm_available = True
        else:
            self.client = None
            self.llm_available = False
            if enable_verification or enable_correction:
                logging.warning("LLM not available, verification disabled")
    
    def verify(self, 
               ocr_text: str,
               image_path: Optional[str] = None,
               context: Optional[Dict] = None) -> Dict[str, any]:
        """
        Verify OCR result using LLM and rules.
        
        Args:
            ocr_text: Text from OCR engine
            image_path: Path to original image (for vision models)
            context: Additional context (expected format, range, etc.)
            
        Returns:
            Dictionary containing:
                - verified_text: Corrected text
                - is_valid: Whether text passes validation
                - confidence: Verification confidence
                - corrections: List of corrections made
                - validation_errors: List of validation errors
        """
        result = {
            'verified_text': ocr_text,
            'is_valid': True,
            'confidence': 1.0,
            'corrections': [],
            'validation_errors': []
        }
        
        # Rule-based validation
        if self.enable_validation:
            validation_result = self._validate_rules(ocr_text, context)
            result['is_valid'] = validation_result['is_valid']
            result['validation_errors'] = validation_result['errors']
        
        # LLM verification (if available and enabled)
        if self.enable_verification and self.llm_available:
            llm_result = self._verify_with_llm(ocr_text, image_path, context)
            if llm_result:
                result['verified_text'] = llm_result['text']
                result['confidence'] = llm_result['confidence']
                result['corrections'] = llm_result['corrections']
        
        return result
    
    def _validate_rules(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        Apply rule-based validation.
        
        Checks:
        - Digit count (for meter readings)
        - Format (numeric, alphanumeric, etc.)
        - Range (min/max values)
        - Pattern matching
        """
        errors = []
        is_valid = True
        
        if not context:
            context = {}
        
        # Check if text is empty
        if not text or not text.strip():
            errors.append("Empty text")
            is_valid = False
            return {'is_valid': is_valid, 'errors': errors}
        
        text = text.strip()
        
        # Check digit count (for meter readings, typically 5-8 digits)
        expected_length = context.get('expected_length', None)
        if expected_length:
            if len(text) != expected_length:
                errors.append(f"Length mismatch: expected {expected_length}, got {len(text)}")
                is_valid = False
        
        # Check if numeric (for meter readings)
        if context.get('numeric_only', True):
            if not text.isdigit():
                errors.append(f"Non-numeric characters found: {text}")
                is_valid = False
        
        # Check range
        min_value = context.get('min_value', None)
        max_value = context.get('max_value', None)
        
        if text.isdigit():
            value = int(text)
            if min_value is not None and value < min_value:
                errors.append(f"Value {value} below minimum {min_value}")
                is_valid = False
            if max_value is not None and value > max_value:
                errors.append(f"Value {value} above maximum {max_value}")
                is_valid = False
        
        # Check pattern
        pattern = context.get('pattern', None)
        if pattern:
            if not re.match(pattern, text):
                errors.append(f"Pattern mismatch: {pattern}")
                is_valid = False
        
        return {'is_valid': is_valid, 'errors': errors}
    
    def _verify_with_llm(self,
                         ocr_text: str,
                         image_path: Optional[str] = None,
                         context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Verify OCR result using LLM.
        
        Uses vision-language model to:
        1. Read the image directly
        2. Compare with OCR result
        3. Suggest corrections if needed
        """
        if not self.llm_available:
            return None
        
        try:
            # Construct prompt
            prompt = self._construct_verification_prompt(ocr_text, context)
            
            # For now, use text-only verification
            # In production, would use vision model with image
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use text model for now
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert OCR verification assistant. Your task is to verify and correct OCR results from meter readings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            verified_text = response.choices[0].message.content.strip()
            
            # Extract corrections
            corrections = []
            if verified_text != ocr_text:
                corrections.append({
                    'original': ocr_text,
                    'corrected': verified_text,
                    'reason': 'LLM correction'
                })
            
            return {
                'text': verified_text,
                'confidence': 0.9,  # High confidence for LLM
                'corrections': corrections
            }
        
        except Exception as e:
            logging.error(f"LLM verification error: {e}")
            return None
    
    def _construct_verification_prompt(self,
                                       ocr_text: str,
                                       context: Optional[Dict] = None) -> str:
        """Construct prompt for LLM verification."""
        if not context:
            context = {}
        
        prompt = f"""Verify the following OCR result from a meter reading:

OCR Result: "{ocr_text}"

"""
        
        if context.get('expected_length'):
            prompt += f"Expected length: {context['expected_length']} digits\n"
        
        if context.get('numeric_only'):
            prompt += "Expected format: Numeric only\n"
        
        prompt += """
Please verify if this reading is correct. If there are any obvious errors (e.g., 'O' instead of '0', 'I' instead of '1'), correct them.

Respond with ONLY the corrected reading, nothing else.
"""
        
        return prompt
