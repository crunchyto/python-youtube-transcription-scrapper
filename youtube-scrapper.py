import os
import logging
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from typing import Optional, List, Dict
import sys
from pytube import YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import argparse
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeTranscriptionScrapper:
    def __init__(self, youtube_api_key: str):
        """Initialize the scrapper with OpenAI and YouTube API clients"""
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize OpenAI client: " + str(e))
            raise
            
        # Initialize YouTube API client
        try:
            self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
            logger.info("YouTube API client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize YouTube API client: " + str(e))
            raise

    def get_video_transcript(self, video_id):
        """
        Fetch transcript for a YouTube video.
        Tries to fetch 'en' transcript first, then 'es' if 'en' is not available.

        Args:
            video_id: YouTube video ID

        Returns:
            Transcript text if successful, None if failed
        """
        try:
            logger.info("Fetching transcript for video ID: " + video_id)
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None

            # Try to fetch English transcript
            try:
                transcript = transcript_list.find_transcript(['en'])
                logger.info("English transcript found")
            except Exception:
                logger.info("English transcript not found, trying Spanish")
                try:
                    transcript = transcript_list.find_transcript(['es'])
                    logger.info("Spanish transcript found")
                except Exception:
                    logger.error("Neither English nor Spanish transcript found")
                    return None

            # Fetch and join transcript text
            transcript_data = transcript.fetch()
            full_transcript = " ".join([entry.text for entry in transcript_data])
            logger.info("Transcript fetched successfully")
            return full_transcript
        except Exception as e:
            logger.error("Error fetching transcript: " + str(e))
            return None

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
            request = self.youtube.playlists().list(
                part="snippet",
                mine=True,
                maxResults=50
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
            return None

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
            videos = []
            request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50
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
            
            logger.info(f"Found {len(videos)} videos in playlist")
            return videos
            
        except HttpError as e:
            logger.error(f"Error fetching playlist videos: {e}")
            return []

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
            return False

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
            request = self.youtube.playlistItems().delete(id=playlist_item_id)
            request.execute()
            logger.info(f"Successfully removed playlist item {playlist_item_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Error removing video from playlist: {e}")
            return False

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
        # Limit length to avoid filesystem issues
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized

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
            
            prompt = """Please analyze this YouTube video transcript and provide:

1. A comprehensive summary of the main points (3-4 paragraphs) with key points as bullets
2. Extract any references to books, articles, YouTube channels, podcasts, websites, tools, or other media mentioned in the video

Format your response as follows:
SUMMARY:
[Your summary here with bullet points for key insights]

RESOURCES:
1. Resource/Reference #1
2. Resource/Reference #2
...

If no resources are mentioned, write "No specific resources mentioned."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes video transcripts."},
                    {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"}
                ],
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            logger.info("Analysis completed successfully")
            
            # Parse the response to separate summary and resources
            if "SUMMARY:" in analysis and "RESOURCES:" in analysis:
                parts = analysis.split("RESOURCES:")
                summary = parts[0].replace("SUMMARY:", "").strip()
                resources = parts[1].strip()
                return {"summary": summary, "resources": resources}
            else:
                # Fallback if format is not followed
                return {"summary": analysis, "resources": "No specific resources mentioned."}
                
        except Exception as e:
            logger.error("Error analyzing transcript: " + str(e))
            return None

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
            
            # Create filename
            safe_title = self.sanitize_filename(video_title)
            output_filename = f"{safe_title}.txt"
            
            # Write to file
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(f"Video Title: {video_title}\n")
                    f.write(f"Video ID: {video_id}\n")
                    f.write(f"Video URL: https://www.youtube.com/watch?v={video_id}\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("SUMMARY\n")
                    f.write("="*50 + "\n\n")
                    f.write(analysis['summary'] + "\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("RESOURCES CITED\n")
                    f.write("="*50 + "\n\n")
                    f.write(analysis['resources'] + "\n\n")
                    
                    f.write("="*50 + "\n")
                    f.write("FULL TRANSCRIPTION\n")
                    f.write("="*50 + "\n\n")
                    f.write(transcript + "\n")
                
                logger.info(f"Successfully saved analysis to {output_filename}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to write file {output_filename}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return False

def main():
    """Main function to process videos from AI_TO_TRANSCRIBE playlist"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YouTube Video Transcription and Analysis with Playlist Management')
    parser.add_argument('youtube_api_key', help='YouTube Data API v3 key')
    parser.add_argument('--source-playlist', default='AI_TO_TRANSCRIBE', 
                        help='Source playlist name (default: AI_TO_TRANSCRIBE)')
    parser.add_argument('--dest-playlist', default='AI_TRANSCRIBED', 
                        help='Destination playlist name (default: AI_TRANSCRIBED)')
    
    args = parser.parse_args()
    
    try:
        # Initialize scrapper
        scrapper = YouTubeTranscriptionScrapper(args.youtube_api_key)
        
        # Get playlist IDs
        source_playlist_id = scrapper.get_playlist_id_by_name(args.source_playlist)
        if not source_playlist_id:
            logger.error(f"Source playlist '{args.source_playlist}' not found")
            return
            
        dest_playlist_id = scrapper.get_playlist_id_by_name(args.dest_playlist)
        if not dest_playlist_id:
            logger.error(f"Destination playlist '{args.dest_playlist}' not found")
            return
        
        # Get videos from source playlist
        videos = scrapper.get_playlist_videos(source_playlist_id)
        if not videos:
            logger.info(f"No videos found in playlist '{args.source_playlist}'")
            return
        
        logger.info(f"Found {len(videos)} videos to process")
        
        # Process each video
        processed_count = 0
        failed_count = 0
        
        for video in videos:
            video_id = video['id']
            video_title = video['title']
            playlist_item_id = video['playlistItemId']
            
            logger.info(f"Processing video {processed_count + failed_count + 1}/{len(videos)}: {video_title}")
            
            # Process the video
            if scrapper.process_video(video_id, video_title):
                # Successfully processed, move to destination playlist
                if scrapper.add_video_to_playlist(video_id, dest_playlist_id):
                    if scrapper.remove_video_from_playlist(playlist_item_id):
                        logger.info(f"Successfully moved video '{video_title}' to {args.dest_playlist}")
                        processed_count += 1
                    else:
                        logger.error(f"Failed to remove video from source playlist: {video_title}")
                        failed_count += 1
                else:
                    logger.error(f"Failed to add video to destination playlist: {video_title}")
                    failed_count += 1
            else:
                logger.error(f"Failed to process video: {video_title}")
                failed_count += 1
        
        # Summary
        logger.info(f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")
        print(f"\nProcessing Summary:")
        print(f"Successfully processed and moved: {processed_count} videos")
        print(f"Failed to process: {failed_count} videos")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return

if __name__ == "__main__":
    main()
