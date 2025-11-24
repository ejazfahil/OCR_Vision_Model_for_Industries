"""
Ensemble OCR Engine integrating multiple SOTA OCR models.
Implements PaddleOCR-VL, TrOCR, and EasyOCR with intelligent voting.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# OCR imports (will be installed via requirements.txt)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.warning("TrOCR not available")


@dataclass
class OCRResult:
    """Container for OCR result with confidence."""
    text: str
    confidence: float
    engine: str
    bbox: Optional[List[Tuple[int, int]]] = None
    

class EnsembleOCR:
    """
    Ensemble OCR system combining multiple engines.
    
    Integrates:
    - PaddleOCR-VL (primary): SOTA lightweight model
    - TrOCR (secondary): Transformer-based for complex text
    - EasyOCR (fallback): Fast and reliable baseline
    
    Uses confidence-weighted voting and majority voting for final prediction.
    """
    
    def __init__(self,
                 use_paddle: bool = True,
                 use_trocr: bool = True,
                 use_easyocr: bool = True,
                 paddle_lang: str = 'en',
                 easyocr_langs: List[str] = ['en'],
                 trocr_model: str = 'microsoft/trocr-base-printed',
                 confidence_threshold: float = 0.6,
                 voting_method: str = 'weighted'):
        """
        Initialize ensemble OCR with configurable engines.
        
        Args:
            use_paddle: Enable PaddleOCR
            use_trocr: Enable TrOCR
            use_easyocr: Enable EasyOCR
            paddle_lang: Language for PaddleOCR
            easyocr_langs: Languages for EasyOCR
            trocr_model: TrOCR model identifier
            confidence_threshold: Minimum confidence for predictions
            voting_method: 'weighted', 'majority', or 'highest'
        """
        self.use_paddle = use_paddle and PADDLE_AVAILABLE
        self.use_trocr = use_trocr and TROCR_AVAILABLE
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.voting_method = voting_method
        
        # Initialize engines
        self.engines = {}
        
        if self.use_paddle:
            try:
                self.engines['paddle'] = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    show_log=False
                )
                logging.info("PaddleOCR initialized")
            except Exception as e:
                logging.error(f"Failed to initialize PaddleOCR: {e}")
                self.use_paddle = False
        
        if self.use_easyocr:
            try:
                self.engines['easyocr'] = easyocr.Reader(
                    easyocr_langs,
                    gpu=True
                )
                logging.info("EasyOCR initialized")
            except Exception as e:
                logging.error(f"Failed to initialize EasyOCR: {e}")
                self.use_easyocr = False
        
        if self.use_trocr:
            try:
                self.engines['trocr_processor'] = TrOCRProcessor.from_pretrained(
                    trocr_model
                )
                self.engines['trocr_model'] = VisionEncoderDecoderModel.from_pretrained(
                    trocr_model
                )
                logging.info("TrOCR initialized")
            except Exception as e:
                logging.error(f"Failed to initialize TrOCR: {e}")
                self.use_trocr = False
    
    def recognize(self, image: np.ndarray) -> Dict[str, any]:
        """
        Perform OCR using ensemble of engines.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Dictionary containing:
                - text: Final recognized text
                - confidence: Confidence score
                - individual_results: Results from each engine
                - voting_details: Details of voting process
        """
        results = []
        
        # Run PaddleOCR
        if self.use_paddle:
            paddle_result = self._run_paddle(image)
            if paddle_result:
                results.append(paddle_result)
        
        # Run EasyOCR
        if self.use_easyocr:
            easy_result = self._run_easyocr(image)
            if easy_result:
                results.append(easy_result)
        
        # Run TrOCR
        if self.use_trocr:
            trocr_result = self._run_trocr(image)
            if trocr_result:
                results.append(trocr_result)
        
        # Perform voting
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'individual_results': [],
                'voting_details': 'No engines produced results'
            }
        
        final_text, final_confidence, voting_details = self._vote(results)
        
        return {
            'text': final_text,
            'confidence': final_confidence,
            'individual_results': results,
            'voting_details': voting_details
        }
    
    def _run_paddle(self, image: np.ndarray) -> Optional[OCRResult]:
        """Run PaddleOCR on image."""
        try:
            result = self.engines['paddle'].ocr(image, cls=True)
            
            if not result or not result[0]:
                return None
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for line in result[0]:
                bbox, (text, conf) = line
                texts.append(text)
                confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                engine='PaddleOCR'
            )
        
        except Exception as e:
            logging.error(f"PaddleOCR error: {e}")
            return None
    
    def _run_easyocr(self, image: np.ndarray) -> Optional[OCRResult]:
        """Run EasyOCR on image."""
        try:
            result = self.engines['easyocr'].readtext(image)
            
            if not result:
                return None
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for bbox, text, conf in result:
                texts.append(text)
                confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                engine='EasyOCR'
            )
        
        except Exception as e:
            logging.error(f"EasyOCR error: {e}")
            return None
    
    def _run_trocr(self, image: np.ndarray) -> Optional[OCRResult]:
        """Run TrOCR on image."""
        try:
            # Convert numpy array to PIL Image
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            # Process image
            pixel_values = self.engines['trocr_processor'](
                pil_image, 
                return_tensors="pt"
            ).pixel_values
            
            # Generate text
            generated_ids = self.engines['trocr_model'].generate(pixel_values)
            generated_text = self.engines['trocr_processor'].batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # TrOCR doesn't provide confidence, use 0.8 as default
            return OCRResult(
                text=generated_text,
                confidence=0.8,
                engine='TrOCR'
            )
        
        except Exception as e:
            logging.error(f"TrOCR error: {e}")
            return None
    
    def _vote(self, results: List[OCRResult]) -> Tuple[str, float, str]:
        """
        Perform voting among OCR results.
        
        Args:
            results: List of OCRResult objects
            
        Returns:
            Tuple of (final_text, final_confidence, voting_details)
        """
        if not results:
            return '', 0.0, 'No results to vote on'
        
        if len(results) == 1:
            return results[0].text, results[0].confidence, 'Single engine'
        
        if self.voting_method == 'highest':
            # Return result with highest confidence
            best = max(results, key=lambda x: x.confidence)
            return best.text, best.confidence, f'Highest confidence: {best.engine}'
        
        elif self.voting_method == 'weighted':
            # Weighted voting based on confidence
            text_scores = {}
            
            for result in results:
                text = result.text.strip()
                if text not in text_scores:
                    text_scores[text] = 0.0
                text_scores[text] += result.confidence
            
            if not text_scores:
                return '', 0.0, 'No valid texts'
            
            best_text = max(text_scores, key=text_scores.get)
            avg_confidence = text_scores[best_text] / len(results)
            
            return best_text, avg_confidence, f'Weighted voting: {len(text_scores)} unique texts'
        
        elif self.voting_method == 'majority':
            # Simple majority voting
            from collections import Counter
            texts = [r.text.strip() for r in results]
            counter = Counter(texts)
            most_common = counter.most_common(1)[0]
            
            # Calculate average confidence for the majority text
            confidences = [r.confidence for r in results if r.text.strip() == most_common[0]]
            avg_confidence = np.mean(confidences)
            
            return most_common[0], avg_confidence, f'Majority: {most_common[1]}/{len(results)} votes'
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def get_active_engines(self) -> List[str]:
        """Return list of active OCR engines."""
        active = []
        if self.use_paddle:
            active.append('PaddleOCR')
        if self.use_easyocr:
            active.append('EasyOCR')
        if self.use_trocr:
            active.append('TrOCR')
        return active
