#!/usr/bin/env python3
"""
Test script to debug Gemini API responses with detailed logging.
Run this with a single item to see what the API actually returns.
"""

import json
import sys
from pathlib import Path
from google import genai
from google.genai import types

def test_gemini_api_response(item_id: str = None):
    """Test Gemini API and log full response structure."""
    
    # Load one item from dataset
    jsonl_path = Path("data/plotchain_v4/plotchain_v4.jsonl")
    if not jsonl_path.exists():
        print(f"❌ Dataset not found: {jsonl_path}")
        return
    
    # Find an item (preferably bandpass since that fails 100%)
    items = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    # Find bandpass item
    test_item = None
    if item_id:
        test_item = next((it for it in items if it['id'] == item_id), None)
    else:
        # Try to find a bandpass item
        test_item = next((it for it in items if it['type'] == 'bandpass_response'), None)
        if not test_item:
            test_item = items[0]  # Fallback to first item
    
    if not test_item:
        print("❌ No test item found")
        return
    
    print("=" * 100)
    print(f"TESTING GEMINI API WITH: {test_item['id']} ({test_item['type']})")
    print("=" * 100)
    
    # Build prompt
    from run_plotchain_v4_eval import build_prompt
    prompt = build_prompt(test_item)
    
    # Get image path
    image_path = Path("data/plotchain_v4/images") / test_item['type'] / f"{test_item['id']}.png"
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\nImage: {image_path}")
    print(f"Prompt length: {len(prompt)} chars")
    
    # Call API
    print("\n" + "=" * 100)
    print("CALLING GEMINI API...")
    print("=" * 100)
    
    try:
        client = genai.Client()
        image_bytes = image_path.read_bytes()
        mime = "image/png"
        
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=8192,  # Maximum allowed for Gemini 2.5 Pro
            ),
        )
        
        print("\n✅ API Call Successful")
        print("=" * 100)
        print("RESPONSE STRUCTURE ANALYSIS")
        print("=" * 100)
        
        # Analyze response
        print(f"\n1. Response Type: {type(resp)}")
        print(f"2. Response Attributes: {[x for x in dir(resp) if not x.startswith('_')]}")
        
        # Check usage_metadata
        if hasattr(resp, 'usage_metadata'):
            usage = resp.usage_metadata
            print(f"\n2.5. Usage Metadata:")
            if hasattr(usage, 'prompt_token_count'):
                print(f"   Prompt tokens: {usage.prompt_token_count}")
            if hasattr(usage, 'candidates_token_count'):
                print(f"   Candidates tokens: {usage.candidates_token_count}")
            if hasattr(usage, 'total_token_count'):
                print(f"   Total tokens: {usage.total_token_count}")
        
        # Check text attribute
        print(f"\n3. Direct 'text' attribute:")
        if hasattr(resp, 'text'):
            text_val = resp.text
            print(f"   Has 'text': Yes")
            print(f"   Value: {repr(text_val)}")
            print(f"   Length: {len(text_val) if text_val else 0}")
        else:
            print(f"   Has 'text': No")
        
        # Check candidates
        print(f"\n4. Candidates:")
        if hasattr(resp, 'candidates'):
            print(f"   Has 'candidates': Yes")
            print(f"   Count: {len(resp.candidates) if resp.candidates else 0}")
            
            if resp.candidates:
                for i, candidate in enumerate(resp.candidates):
                    print(f"\n   Candidate {i}:")
                    print(f"     Type: {type(candidate)}")
                    print(f"     Attributes: {[x for x in dir(candidate) if not x.startswith('_')]}")
                    
                    # Check finish_reason
                    if hasattr(candidate, 'finish_reason'):
                        print(f"     finish_reason: {candidate.finish_reason}")
                    
                    # Check content
                    if hasattr(candidate, 'content'):
                        content = candidate.content
                        print(f"     Has 'content': Yes")
                        print(f"     Content type: {type(content)}")
                        print(f"     Content attributes: {[x for x in dir(content) if not x.startswith('_')]}")
                        
                        # Check parts
                        if hasattr(content, 'parts'):
                            print(f"     Has 'parts': Yes")
                            print(f"     Parts count: {len(content.parts) if content.parts else 0}")
                            
                            for j, part in enumerate(content.parts if content.parts else []):
                                print(f"\n       Part {j}:")
                                print(f"         Type: {type(part)}")
                                print(f"         Attributes: {[x for x in dir(part) if not x.startswith('_')]}")
                                
                                if hasattr(part, 'text'):
                                    part_text = part.text
                                    print(f"         Has 'text': Yes")
                                    print(f"         Text: {repr(part_text[:200])}")
                                    print(f"         Length: {len(part_text) if part_text else 0}")
                                else:
                                    print(f"         Has 'text': No")
                        
                        elif hasattr(content, 'text'):
                            content_text = content.text
                            print(f"     Has direct 'text': Yes")
                            print(f"     Text: {repr(content_text[:200])}")
                            print(f"     Length: {len(content_text) if content_text else 0}")
                        else:
                            print(f"     No 'parts' or 'text' found")
                    else:
                        print(f"     Has 'content': No")
        else:
            print(f"   Has 'candidates': No")
        
        # Try to extract text using our current method
        print("\n" + "=" * 100)
        print("TEXT EXTRACTION ATTEMPT")
        print("=" * 100)
        
        text = ""
        
        # Method 1
        if hasattr(resp, 'text') and resp.text:
            text = resp.text
            print(f"✅ Method 1 (resp.text): Success - {len(text)} chars")
        else:
            print(f"❌ Method 1 (resp.text): Failed")
        
        # Method 2
        if not text and hasattr(resp, 'candidates') and resp.candidates:
            for candidate in resp.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text += part.text
                    elif hasattr(candidate.content, 'text'):
                        text += candidate.content.text
        
        if text:
            print(f"✅ Method 2 (candidates->content->parts): Success - {len(text)} chars")
            print(f"\nExtracted text:\n{text[:500]}")
        else:
            print(f"❌ Method 2 (candidates->content->parts): Failed")
        
        # Save full response to file
        debug_file = Path("gemini_api_response_debug.json")
        debug_data = {
            "item_id": test_item['id'],
            "item_type": test_item['type'],
            "response_type": str(type(resp)),
            "has_text": hasattr(resp, 'text'),
            "text_value": getattr(resp, 'text', None),
            "has_candidates": hasattr(resp, 'candidates'),
            "candidates_count": len(resp.candidates) if hasattr(resp, 'candidates') and resp.candidates else 0,
            "extracted_text": text,
            "extracted_text_length": len(text),
        }
        
        # Add usage metadata if available
        if hasattr(resp, 'usage_metadata'):
            usage = resp.usage_metadata
            debug_data["usage_metadata"] = {
                "prompt_token_count": getattr(usage, 'prompt_token_count', None),
                "candidates_token_count": getattr(usage, 'candidates_token_count', None),
                "total_token_count": getattr(usage, 'total_token_count', None),
            }
        
        if hasattr(resp, 'candidates') and resp.candidates:
            candidates_data = []
            for i, candidate in enumerate(resp.candidates):
                cand_data = {
                    "index": i,
                    "type": str(type(candidate)),
                    "finish_reason": getattr(candidate, 'finish_reason', None),
                    "token_count": getattr(candidate, 'token_count', None),
                    "has_content": hasattr(candidate, 'content'),
                }
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    cand_data["content_type"] = str(type(content))
                    cand_data["has_parts"] = hasattr(content, 'parts')
                    if hasattr(content, 'parts'):
                        if content.parts is not None:
                            parts_data = []
                            for j, part in enumerate(content.parts):
                                part_data = {
                                    "index": j,
                                    "type": str(type(part)),
                                    "has_text": hasattr(part, 'text'),
                                    "text": getattr(part, 'text', None),
                                }
                                parts_data.append(part_data)
                            cand_data["parts"] = parts_data
                        else:
                            cand_data["parts"] = None
                candidates_data.append(cand_data)
            debug_data["candidates"] = candidates_data
        
        with open(debug_file, "w") as f:
            json.dump(debug_data, f, indent=2, default=str)
        
        print(f"\n✅ Full response saved to: {debug_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    item_id = sys.argv[1] if len(sys.argv) > 1 else None
    test_gemini_api_response(item_id)

