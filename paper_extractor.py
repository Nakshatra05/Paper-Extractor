import pandas as pd
import json
import re
from typing import Dict, Union, Any, Optional
import requests
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define schema with data types
SCHEMA = {
    "Pub_year": "int",  # Publication year
    "Journal": "str",  # Name of journal
    "Study_type": "str",  # Hut trial, lab based bioassay, or village trial
    "Net_type": "str",  # The names of the LLINs tested
    "Source": "str",  # 'Wild' or 'Lab'
    "Country": "str",  # Country where the study was conducted
    "Site": "str",  # Specific geographic information
    "Start_date": "str",  # YYYY-MM format
    "End_date": "str",  # YYYY-MM format
    "Time_elapsed": "float"  # Time elapsed in months
}

def fetch_paper_abstract(doi: str) -> Optional[str]:
    """
    Fetch the abstract of a research paper given its DOI.
    
    Args:
        doi: Digital Object Identifier of the paper
    
    Returns:
        The abstract text or None if it couldn't be retrieved
    """
    try:
        # Try to fetch from doi.org
        url = f"https://doi.org/{doi}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML and extract abstract
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Different publishers format their pages differently, so this is a simplistic approach
        # In a real application, you might need more sophisticated parsing logic
        abstract_section = soup.find('section', {'id': 'Abs1'}) or soup.find('div', {'class': 'abstract'})
        
        if abstract_section:
            return abstract_section.get_text().strip()
        else:
            logger.warning(f"Could not find abstract section in HTML for DOI: {doi}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching abstract for DOI {doi}: {str(e)}")
        return None

def create_extraction_prompt(abstract: str) -> str:
    """
    Create a prompt for the LLM to extract structured data from an abstract.
    
    Args:
        abstract: The paper abstract text
    
    Returns:
        A formatted prompt string
    """
    prompt = f"""
Extract the following structured information from this research paper abstract:

Pub_year: Publication year as an integer (e.g., 2019)
Journal: Name of journal as a string
Study_type: One of "Hut trial", "Lab based bioassay", or "Village trial"
Net_type: Names of long-lasting insecticide-treated nets (LLINs) tested, comma-separated if multiple
Source: Whether mosquitoes were from the wild or lab strain, answer only "Wild" or "Lab"
Country: Country where the study was conducted
Site: Specific geographic location information (e.g., district name)
Start_date: Study start date in YYYY-MM format
End_date: Study end date in YYYY-MM format
Time_elapsed: Time since start of study in months (as a number)

Abstract: 
{abstract}

Format your response as a valid JSON object with these exact field names.
Only include fields where information is clearly provided in the abstract.
For fields where information is not available, use null.
"""
    return prompt

def extract_with_openai(abstract: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Extract data from abstract using OpenAI.
    
    Args:
        abstract: Paper abstract text
        api_key: OpenAI API key
        
    Returns:
        Dictionary containing extracted data or None if extraction failed
    """
    try:
        import openai
        openai.api_key = api_key
        
        prompt = create_extraction_prompt(abstract)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from research papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        
        # Parse the JSON
        extracted_data = json.loads(result_text)
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error in OpenAI extraction: {str(e)}")
        return None

def extract_with_llamaindex(abstract: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extract data from abstract using LlamaIndex with a structured output parser.
    
    Args:
        abstract: Paper abstract text
        api_key: Optional API key for the LLM service
        
    Returns:
        Dictionary containing extracted data or None if extraction failed
    """
    try:
        from llama_index.program import OpenAIPydanticProgram
        from pydantic import BaseModel, Field
        from typing import Optional as OptField
        import os
        
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Define Pydantic model for structured output
        class ResearchData(BaseModel):
            Pub_year: OptField[int] = Field(description="Publication year as an integer")
            Journal: OptField[str] = Field(description="Name of journal")
            Study_type: OptField[str] = Field(description="One of 'Hut trial', 'Lab based bioassay', or 'Village trial'")
            Net_type: OptField[str] = Field(description="Names of LLINs tested, comma-separated if multiple")
            Source: OptField[str] = Field(description="Whether mosquitoes were from the wild or lab, answer 'Wild' or 'Lab'")
            Country: OptField[str] = Field(description="Country where the study was conducted")
            Site: OptField[str] = Field(description="Specific geographic location information")
            Start_date: OptField[str] = Field(description="Study start date in YYYY-MM format")
            End_date: OptField[str] = Field(description="Study end date in YYYY-MM format")
            Time_elapsed: OptField[float] = Field(description="Time since start of study in months")
        
        # Create the extraction program
        prompt_template = """
        Extract the following structured information from this research paper abstract.
        Only include fields where information is clearly provided in the abstract.
        
        Abstract:
        {abstract}
        """
        
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=ResearchData,
            prompt_template_str=prompt_template,
            verbose=True
        )
        
        # Run the extraction
        result = program(abstract=abstract)
        
        # Convert Pydantic model to dictionary
        extracted_data = result.dict()
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error in LlamaIndex extraction: {str(e)}")
        return None

def validate_and_convert_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and convert extracted data to match the schema data types.
    
    Args:
        data: Dictionary of extracted data
    
    Returns:
        Dictionary with values converted to appropriate types
    """
    validated_data = {}
    
    for field, dtype in SCHEMA.items():
        value = data.get(field)
        
        # Skip null/None values
        if value is None:
            validated_data[field] = None
            continue
            
        try:
            if dtype == "int":
                # Try to extract year from a string like "Published in 2019"
                if isinstance(value, str):
                    year_match = re.search(r'\b(19|20)\d{2}\b', value)
                    if year_match:
                        validated_data[field] = int(year_match.group(0))
                    else:
                        validated_data[field] = int(value)
                else:
                    validated_data[field] = int(value)
                    
            elif dtype == "float":
                validated_data[field] = float(value)
                
            elif dtype == "str":
                validated_data[field] = str(value).strip()
                
            else:
                validated_data[field] = value
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert {field} value '{value}' to {dtype}: {str(e)}")
            validated_data[field] = None
            
    return validated_data

def extract_data_from_abstract(abstract: str, llm_service: str = "openai", api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Extract structured data from a research paper abstract using an LLM.
    
    Args:
        abstract: The research paper abstract text
        llm_service: The LLM service to use ('openai' or 'llamaindex')
        api_key: API key for the LLM service
        
    Returns:
        A pandas DataFrame containing the extracted data
    """
    if not abstract:
        logger.error("No abstract text provided")
        return pd.DataFrame(columns=SCHEMA.keys())
    
    # Extract data using the specified LLM service
    if llm_service.lower() == "openai":
        extracted_data = extract_with_openai(abstract, api_key)
    elif llm_service.lower() == "llamaindex":
        extracted_data = extract_with_llamaindex(abstract, api_key)
    else:
        logger.error(f"Unsupported LLM service: {llm_service}")
        return pd.DataFrame(columns=SCHEMA.keys())
    
    # Handle extraction failure
    if not extracted_data:
        logger.error("Data extraction failed")
        return pd.DataFrame(columns=SCHEMA.keys())
    
    # Validate and convert data types
    validated_data = validate_and_convert_types(extracted_data)
    
    # Convert to DataFrame
    df = pd.DataFrame([validated_data])
    return df

def extract_data_from_doi(doi: str, llm_service: str = "openai", api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch a paper abstract by DOI and extract structured data from it.
    
    Args:
        doi: The paper's DOI
        llm_service: The LLM service to use ('openai' or 'llamaindex')
        api_key: API key for the LLM service
        
    Returns:
        A pandas DataFrame containing the extracted data
    """
    abstract = fetch_paper_abstract(doi)
    
    if not abstract:
        logger.error(f"Could not fetch abstract for DOI: {doi}")
        return pd.DataFrame(columns=SCHEMA.keys())
    
    return extract_data_from_abstract(abstract, llm_service, api_key)

# Example usage
if __name__ == "__main__":
    # Example DOI from the task
    example_doi = "10.1186/s12936-019-2973-x"
    
    # For testing without API key, you can use this hardcoded abstract
    example_abstract = """
    Background: Long-lasting insecticidal nets (LLINs) are the primary malaria prevention approach globally. However, 
    insecticide resistance in vectors threatens the efficacy of insecticidal interventions, including LLINs. 
    Interceptor® G2 is a new LLIN that contains a mixture of two insecticides: alpha-cypermethrin and chlorfenapyr.
    
    Methods: This study was conducted in Northeast Tanzania between December 2017 to January 2018. The efficacy of unwashed 
    and 20-times washed Interceptor G2 LLIN was evaluated in both World Health Organization tunnel tests and experimental hut 
    trials, and compared to standard Interceptor LLIN (treated with alpha-cypermethrin only) and an untreated control net. 
    The main vectors at the site were resistant to pyrethroids and susceptible to chlorfenapyr. Free-flying Anopheles 
    funestus sensu lato and Anopheles gambiae sensu lato were collected each morning and scored for 24-h mortality, blood 
    feeding inhibition and exophily.
    
    Results: In tunnel tests, 20-times washing of Interceptor G2 reduced chlorfenapyr content by only 25%, while the pyrethroid 
    content was reduced by 71% and the mortality of resistant Anopheles arabiensis was 24% after 20 washes. In hut trials, 
    Interceptor G2 killed 71% of wild An. funestus s.l. when unwashed and 51% when washed 20 times. Against An. gambiae s.l., 
    unwashed and washed Interceptor G2 killed 61% and 49%, respectively. Interceptor G2 demonstrated superior wash retention 
    to standard Interceptor LLIN, and both blood feeding inhibition and exophily were similar for Interceptor G2 and Interceptor.
    
    Conclusion: Interceptor G2 was superior at killing of pyrethroid resistant An. funestus s.l. and An. gambiae s.l. compared to 
    standard Interceptor LLIN while having similar excito-repellent properties. Based on 24-h mortality rates, this study 
    confirms the promising performance of Interceptor G2 for controlling pyrethroid-resistant mosquito vectors.
    """
    
    print("Testing extraction function with example abstract:")
    
    # Replace 'your_api_key' with an actual API key if using OpenAI
    # Or set the environment variable OPENAI_API_KEY
    # df = extract_data_from_abstract(example_abstract, llm_service="openai", api_key="your_api_key")
    
    # For demonstration, you could print the expected extraction manually:
    expected_data = {
        "Pub_year": 2019,
        "Journal": "Malaria Journal",  # Based on the DOI
        "Study_type": "Hut trial",
        "Net_type": "Interceptor G2, Interceptor",
        "Source": "Wild",
        "Country": "Tanzania",
        "Site": "Northeast Tanzania",
        "Start_date": "2017-12",
        "End_date": "2018-01",
        "Time_elapsed": 1.0  # approximately 1 month
    }
    
    print("\nExpected extraction result:")
    print(pd.DataFrame([expected_data]))
    
    print("\nTo use this code with your own API key:")
    print("1. Import the module: from paper_extractor import extract_data_from_abstract")
    print("2. Call the function with your abstract and API key:")
    print("   df = extract_data_from_abstract(abstract_text, api_key='your_api_key')") 