import os
import logging
import asyncio
import time
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from typing import Optional, List, Dict
import sys
from pytube import YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth
import pickle
import argparse
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import aiohttp
from config import CONFIG


# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG.logging.level),
    format=CONFIG.logging.format,
    filename=CONFIG.logging.file_output
)
logger = logging.getLogger(__name__)

class YouTubeTranscriptionScrapper:
    def __init__(self, youtube_api_key: Optional[str] = None, use_oauth: bool = False):
        """Initialize the scrapper with OpenAI and YouTube API clients"""
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Initialize YouTube API client
        if use_oauth:
            self._init_youtube_oauth_client()
        else:
            self._init_youtube_api_client(youtube_api_key)
            
        # Rate limiting tracking
        self.last_api_call = 0
        self.api_call_count = 0
        self.rate_limit_window_start = time.time()
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            self.client = OpenAI(api_key=openai_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize OpenAI client: " + str(e))
            raise
    
    def _init_youtube_api_client(self, youtube_api_key: str):
        """Initialize YouTube API client with API key"""
        try:
            if not youtube_api_key:
                raise ValueError("YouTube API key is required")
            
            self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
            logger.info("YouTube API client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize YouTube API client: " + str(e))
            raise
    
    def _init_youtube_oauth_client(self):
        """Initialize YouTube API client with OAuth authentication"""
        SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
        creds = None
        
        # Check if token.pickle exists
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('client_secrets.json'):
                    raise FileNotFoundError(
                        "client_secrets.json not found. Please download it from Google Cloud Console."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secrets.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        try:
            self.youtube = build('youtube', 'v3', credentials=creds)
            logger.info("YouTube OAuth client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize YouTube OAuth client: " + str(e))
            raise
    
    
    def _rate_limit_check(self) -> None:
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.rate_limit_window_start >= 60:
            self.api_call_count = 0
            self.rate_limit_window_start = current_time
        
        # Check if we're hitting rate limits
        if self.api_call_count >= CONFIG.api.rate_limit_requests_per_minute:
            sleep_time = 60 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                self.api_call_count = 0
                self.rate_limit_window_start = time.time()
        
        self.api_call_count += 1
        self.last_api_call = current_time

    @retry(
        stop=stop_after_attempt(CONFIG.api.max_retries),
        wait=wait_exponential(multiplier=CONFIG.api.retry_delay, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def get_video_transcript(self, video_id: str) -> Optional[str]:
        """
        Fetch transcript for a YouTube video with retry logic.
        Tries preferred languages in order.

        Args:
            video_id: YouTube video ID

        Returns:
            Transcript text if successful, None if failed
        """
        try:
            logger.info(f"Fetching transcript for video ID: {video_id}")
            self._rate_limit_check()
            
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None

            # Try preferred languages in order
            for lang in CONFIG.transcript.preferred_languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    logger.info(f"{lang.upper()} transcript found")
                    break
                except Exception:
                    logger.debug(f"{lang.upper()} transcript not found")
                    continue
            
            if not transcript:
                logger.error(f"No transcript found in preferred languages: {CONFIG.transcript.preferred_languages}")
                return None

            # Fetch and join transcript text
            transcript_data = transcript.fetch()
            full_transcript = " ".join([entry.text for entry in transcript_data])
            
            # Check transcript length and truncate if necessary
            if len(full_transcript) > CONFIG.transcript.max_transcript_length:
                logger.warning(f"Transcript too long ({len(full_transcript)} chars), truncating to {CONFIG.transcript.max_transcript_length}")
                full_transcript = full_transcript[:CONFIG.transcript.max_transcript_length]
            
            logger.info("Transcript fetched successfully")
            return full_transcript
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(CONFIG.api.max_retries),
        wait=wait_exponential(multiplier=CONFIG.api.retry_delay, min=1, max=10),
        retry=retry_if_exception_type((HttpError,))
    )
    def get_playlist_id_by_name(self, playlist_name: str) -> Optional[str]:
        """
        Get playlist ID by playlist name from the authenticated user's playlists
        
        Args:
            playlist_name: Name of the playlist to find
            
        Returns:
            Playlist ID if found, None otherwise
        """
        try:
            logger.info(f"Searching for playlist: {playlist_name}")
            self._rate_limit_check()
            
            request = self.youtube.playlists().list(
                part="snippet",
                mine=True,
                maxResults=CONFIG.api.youtube_api_max_results
            )
            response = request.execute()
            
            for playlist in response.get('items', []):
                if playlist['snippet']['title'] == playlist_name:
                    playlist_id = playlist['id']
                    logger.info(f"Found playlist '{playlist_name}' with ID: {playlist_id}")
                    return playlist_id
            
            logger.warning(f"Playlist '{playlist_name}' not found")
            return None
            
        except HttpError as e:
            logger.error(f"Error searching for playlist '{playlist_name}': {e}")
            raise

    @retry(
        stop=stop_after_attempt(CONFIG.api.max_retries),
        wait=wait_exponential(multiplier=CONFIG.api.retry_delay, min=1, max=10),
        retry=retry_if_exception_type((HttpError,))
    )
    def get_playlist_videos(self, playlist_id: str) -> List[Dict]:
        """
        Get all videos from a playlist
        
        Args:
            playlist_id: YouTube playlist ID
            
        Returns:
            List of video dictionaries with id, title, and other metadata
        """
        try:
            logger.info(f"Fetching videos from playlist: {playlist_id}")
            self._rate_limit_check()
            
            videos = []
            request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=CONFIG.api.youtube_api_max_results
            )
            
            while request:
                response = request.execute()
                for item in response.get('items', []):
                    video = {
                        'id': item['snippet']['resourceId']['videoId'],
                        'title': item['snippet']['title'],
                        'playlistItemId': item['id']
                    }
                    videos.append(video)
                
                request = self.youtube.playlistItems().list_next(request, response)
                if request:
                    self._rate_limit_check()
            
            logger.info(f"Found {len(videos)} videos in playlist")
            return videos
            
        except HttpError as e:
            logger.error(f"Error fetching playlist videos: {e}")
            raise

    @retry(
        stop=stop_after_attempt(CONFIG.api.max_retries),
        wait=wait_exponential(multiplier=CONFIG.api.retry_delay, min=1, max=10),
        retry=retry_if_exception_type((HttpError,))
    )
    def add_video_to_playlist(self, video_id: str, playlist_id: str) -> bool:
        """
        Add a video to a playlist
        
        Args:
            video_id: YouTube video ID
            playlist_id: Target playlist ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Adding video {video_id} to playlist {playlist_id}")
            self._rate_limit_check()
            
            request = self.youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }
            )
            request.execute()
            logger.info(f"Successfully added video {video_id} to playlist")
            return True
            
        except HttpError as e:
            logger.error(f"Error adding video to playlist: {e}")
            raise

    @retry(
        stop=stop_after_attempt(CONFIG.api.max_retries),
        wait=wait_exponential(multiplier=CONFIG.api.retry_delay, min=1, max=10),
        retry=retry_if_exception_type((HttpError,))
    )
    def remove_video_from_playlist(self, playlist_item_id: str) -> bool:
        """
        Remove a video from a playlist using its playlist item ID
        
        Args:
            playlist_item_id: The playlist item ID (not the video ID)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Removing playlist item {playlist_item_id}")
            self._rate_limit_check()
            
            request = self.youtube.playlistItems().delete(id=playlist_item_id)
            request.execute()
            logger.info(f"Successfully removed playlist item {playlist_item_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Error removing video from playlist: {e}")
            raise

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem
        """
        # Remove invalid characters for filenames
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')
        # Remove extra underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Strip leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to avoid filesystem issues
        if len(sanitized) > CONFIG.file.max_filename_length:
            sanitized = sanitized[:CONFIG.file.max_filename_length]
        return sanitized or "unnamed_video"

    def _chunk_transcript(self, transcript: str) -> List[str]:
        """
        Split large transcripts into manageable chunks
        
        Args:
            transcript: Full transcript text
            
        Returns:
            List of transcript chunks
        """
        if len(transcript) <= CONFIG.transcript.chunk_size:
            return [transcript]
        
        chunks = []
        words = transcript.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > CONFIG.transcript.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @retry(
        stop=stop_after_attempt(CONFIG.api.max_retries),
        wait=wait_exponential(multiplier=CONFIG.api.retry_delay, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def analyze_transcript(self, transcript: str) -> Optional[Dict[str, str]]:
        """
        Analyze transcript using OpenAI to create summary and extract resources
        
        Args:
            transcript: Transcript text
            
        Returns:
            Dictionary with 'summary' and 'resources' keys if successful, None if failed
        """
        try:
            logger.info("Analyzing transcript with OpenAI")
            self._rate_limit_check()
            
            # Handle large transcripts by chunking
            chunks = self._chunk_transcript(transcript)
            
            if len(chunks) == 1:
                # Single chunk - process normally
                return self._analyze_single_transcript(transcript)
            else:
                # Multiple chunks - process each and combine
                logger.info(f"Processing transcript in {len(chunks)} chunks")
                return self._analyze_chunked_transcript(chunks)
                
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            raise
    
    def _analyze_single_transcript(self, transcript: str) -> Dict[str, str]:
        """Analyze a single transcript chunk"""
        prompt = """Please analyze this YouTube video transcript and provide:

1. A comprehensive summary of the main points (2-3 paragraphs)
2. Key takeaways as bullet points
3. Extract any references to YouTube videos, podcasts, websites, URLs mentioned in the video
4. Extract any references to books, articles, tools, software, or other media mentioned in the video

Format your response as follows:
SUMMARY:
[Your comprehensive summary here]

KEY POINTS:
• Key point #1
• Key point #2
• Key point #3
...

REFERENCES:
1. YouTube Video/Channel: [name/link]
2. Podcast: [name/link]
3. Website/URL: [name/link]
...

RESOURCES:
1. Book: [title and author]
2. Tool/Software: [name]
3. Article: [title]
...

If no references are mentioned, write "No specific references mentioned."
If no resources are mentioned, write "No specific resources mentioned."""

        response = self.client.chat.completions.create(
            model=CONFIG.api.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes video transcripts."},
                {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"}
            ],
            temperature=CONFIG.api.openai_temperature,
            max_tokens=CONFIG.api.openai_max_tokens
        )
        
        analysis = response.choices[0].message.content
        logger.info("Analysis completed successfully")
        
        # Parse the response to separate summary, key points, references, and resources
        sections = {"summary": "", "key_points": "", "references": "", "resources": ""}
        
        if "SUMMARY:" in analysis:
            # Extract summary
            if "KEY POINTS:" in analysis:
                summary_part = analysis.split("KEY POINTS:")[0].replace("SUMMARY:", "").strip()
            else:
                summary_part = analysis.replace("SUMMARY:", "").strip()
            sections["summary"] = summary_part
            
            # Extract key points
            if "KEY POINTS:" in analysis and "REFERENCES:" in analysis:
                key_points_part = analysis.split("KEY POINTS:")[1].split("REFERENCES:")[0].strip()
                sections["key_points"] = key_points_part
            elif "KEY POINTS:" in analysis:
                key_points_part = analysis.split("KEY POINTS:")[1].strip()
                sections["key_points"] = key_points_part
            else:
                sections["key_points"] = "No key points extracted."
                
            # Extract references
            if "REFERENCES:" in analysis and "RESOURCES:" in analysis:
                references_part = analysis.split("REFERENCES:")[1].split("RESOURCES:")[0].strip()
                sections["references"] = references_part
            elif "REFERENCES:" in analysis:
                references_part = analysis.split("REFERENCES:")[1].strip()
                sections["references"] = references_part
            else:
                sections["references"] = "No specific references mentioned."
                
            # Extract resources
            if "RESOURCES:" in analysis:
                resources_part = analysis.split("RESOURCES:")[1].strip()
                sections["resources"] = resources_part
            else:
                sections["resources"] = "No specific resources mentioned."
                
            return sections
        else:
            # Fallback if format is not followed
            return {
                "summary": analysis,
                "key_points": "No key points extracted.",
                "references": "No specific references mentioned.",
                "resources": "No specific resources mentioned."
            }
    
    def _analyze_chunked_transcript(self, chunks: List[str]) -> Dict[str, str]:
        """Analyze multiple transcript chunks and create unified analysis"""
        # Combine all chunks into one full transcript for unified analysis
        full_transcript = "\n\n".join(chunks)
        
        # Create a unified analysis of the complete transcript
        logger.info(f"Creating unified analysis from {len(chunks)} chunks")
        unified_analysis = self._analyze_single_transcript(full_transcript)
        
        return unified_analysis

    def process_video(self, video_id: str, video_title: str) -> bool:
        """
        Process a single video: get transcript, analyze it, and save to file
        
        Args:
            video_id: YouTube video ID
            video_title: Video title for filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing video: {video_title} ({video_id})")
            
            # Get transcript
            transcript = self.get_video_transcript(video_id)
            if not transcript:
                logger.error(f"Failed to get transcript for video {video_id}")
                return False
            
            # Analyze transcript
            analysis = self.analyze_transcript(transcript)
            if not analysis:
                logger.error(f"Failed to analyze transcript for video {video_id}")
                return False
            
            # Create filename and output path
            safe_title = self.sanitize_filename(video_title)
            output_filename = f"{safe_title}.txt"
            output_path = os.path.join(CONFIG.file.output_directory, output_filename)
            
            # Ensure output directory exists
            os.makedirs(CONFIG.file.output_directory, exist_ok=True)
            
            # Write to file
            try:
                with open(output_path, "w", encoding=CONFIG.file.file_encoding) as f:
                    f.write(f"Video Title: {video_title}\n")
                    f.write(f"Video ID: {video_id}\n")
                    f.write(f"Video URL: https://www.youtube.com/watch?v={video_id}\n")
                    f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("FULL TRANSCRIPTION\n")
                    f.write("="*50 + "\n\n")
                    f.write(transcript + "\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("SUMMARY\n")
                    f.write("="*50 + "\n\n")
                    f.write(analysis['summary'] + "\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("KEY POINTS\n")
                    f.write("="*50 + "\n\n")
                    f.write(analysis['key_points'] + "\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("REFERENCES\n")
                    f.write("="*50 + "\n\n")
                    f.write(analysis['references'] + "\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("RESOURCES CITED\n")
                    f.write("="*50 + "\n\n")
                    f.write(analysis['resources'] + "\n")
                
                logger.info(f"Successfully saved analysis to {output_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to write file {output_path}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return False

def main():
    """Main function to process videos from AI_TO_TRANSCRIBE playlist"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YouTube Video Transcription and Analysis with Playlist Management')
    parser.add_argument('--youtube-api-key', help='YouTube Data API v3 key (for public videos only)')
    parser.add_argument('--oauth', action='store_true', 
                        help='Use OAuth authentication (required for private playlists)')
    parser.add_argument('--source-playlist', default=CONFIG.playlist.default_source_playlist, 
                        help=f'Source playlist name (default: {CONFIG.playlist.default_source_playlist})')
    parser.add_argument('--dest-playlist', default=CONFIG.playlist.default_dest_playlist, 
                        help=f'Destination playlist name (default: {CONFIG.playlist.default_dest_playlist})')
    parser.add_argument('--output-dir', default=CONFIG.file.output_directory,
                        help=f'Output directory for transcript files (default: {CONFIG.file.output_directory})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process videos but do not move them between playlists')
    
    args = parser.parse_args()
    
    # Validate authentication arguments
    if not args.oauth and not args.youtube_api_key:
        parser.error("Either --oauth or --youtube-api-key must be specified")
    
    # Update config with command line arguments
    CONFIG.file.output_directory = args.output_dir
    
    try:
        # Initialize scrapper
        logger.info("Initializing YouTube Transcription Scrapper...")
        if args.oauth:
            logger.info("Using OAuth authentication for YouTube API")
            scrapper = YouTubeTranscriptionScrapper(use_oauth=True)
        else:
            logger.info("Using API key for YouTube API")
            scrapper = YouTubeTranscriptionScrapper(youtube_api_key=args.youtube_api_key)
        
        # Get playlist IDs
        logger.info(f"Looking up playlist IDs...")
        source_playlist_id = scrapper.get_playlist_id_by_name(args.source_playlist)
        if not source_playlist_id:
            logger.error(f"Source playlist '{args.source_playlist}' not found")
            return
            
        dest_playlist_id = scrapper.get_playlist_id_by_name(args.dest_playlist)
        if not dest_playlist_id:
            logger.error(f"Destination playlist '{args.dest_playlist}' not found")
            return
        
        # Get videos from source playlist
        logger.info(f"Fetching videos from '{args.source_playlist}' playlist...")
        videos = scrapper.get_playlist_videos(source_playlist_id)
        if not videos:
            logger.info(f"No videos found in playlist '{args.source_playlist}'")
            return
        
        logger.info(f"Found {len(videos)} videos to process")
        
        if args.dry_run:
            logger.info("DRY RUN MODE: Videos will be processed but not moved between playlists")
        
        # Process each video with progress bar
        processed_count = 0
        failed_count = 0
        
        with tqdm(total=len(videos), desc="Processing videos", unit="video") as pbar:
            for video in videos:
                video_id = video['id']
                video_title = video['title']
                playlist_item_id = video['playlistItemId']
                
                pbar.set_description(f"Processing: {video_title[:50]}{'...' if len(video_title) > 50 else ''}")
                logger.info(f"Processing video {processed_count + failed_count + 1}/{len(videos)}: {video_title}")
                
                try:
                    # Process the video
                    if scrapper.process_video(video_id, video_title):
                        if not args.dry_run:
                            # Successfully processed, move to destination playlist
                            try:
                                scrapper.add_video_to_playlist(video_id, dest_playlist_id)
                                scrapper.remove_video_from_playlist(playlist_item_id)
                                logger.info(f"Successfully moved video '{video_title}' to {args.dest_playlist}")
                            except Exception as e:
                                logger.error(f"Failed to move video '{video_title}': {e}")
                                failed_count += 1
                                pbar.update(1)
                                continue
                        
                        processed_count += 1
                        pbar.set_postfix({"✓": processed_count, "✗": failed_count})
                    else:
                        logger.error(f"Failed to process video: {video_title}")
                        failed_count += 1
                        pbar.set_postfix({"✓": processed_count, "✗": failed_count})
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing video '{video_title}': {e}")
                    failed_count += 1
                    pbar.set_postfix({"✓": processed_count, "✗": failed_count})
                
                pbar.update(1)
        
        # Summary
        logger.info(f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")
        print(f"\n{'='*50}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Successfully processed: {processed_count} videos")
        print(f"Failed to process: {failed_count} videos")
        print(f"Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%" if (processed_count + failed_count) > 0 else "No videos processed")
        print(f"Output directory: {CONFIG.file.output_directory}")
        if args.dry_run:
            print(f"NOTE: Dry run mode - videos were not moved between playlists")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\nProcessing interrupted. Partial results may be available in output directory.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\nFatal error: {e}")
        return

if __name__ == "__main__":
    main()
