import os
import asyncio
import json
from xml.dom.minidom import Comment

import numpy as np
import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telethon import TelegramClient, events
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from telethon import functions, types

# Load environment variables
load_dotenv()

# Configuration
API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
SESSION_FILE = os.getenv('SESSION_FILE', 'telegram_session')
DATABASE_URL = os.getenv('DATABASE_URL').replace('sqlite:///', 'sqlite+aiosqlite:///')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SIMILARITY_THRESHOLD = 0.7
DESTINATION_CHANNEL = os.getenv('DESTINATION_CHANNEL')
BOT_TOKEN = os.getenv('BOT_TOKEN')
CLUSTER_TIME_WINDOW = 10  # Hours

# Set up async database
async_engine = create_async_engine(DATABASE_URL)
async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize the Telegram client
telegram_client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
bot = TelegramClient('bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0

    embedding1_array = np.array(embedding1)
    embedding2_array = np.array(embedding2)

    dot_product = np.dot(embedding1_array, embedding2_array)
    norm1 = np.linalg.norm(embedding1_array)
    norm2 = np.linalg.norm(embedding2_array)

    return dot_product / (norm1 * norm2)

async def get_embedding(message_text):
    """Get embedding for text using OpenAI API."""
    try:
        response = await client.embeddings.create(
            input=message_text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

async def find_similar_cluster(session, message_embedding):
    """Find the most similar cluster for a message embedding within time window."""
    # Calculate the cutoff time (current time - 10 hours)
    cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=CLUSTER_TIME_WINDOW)

    query = text("""
    SELECT id, embedding FROM clusters
    WHERE embedding IS NOT NULL 
    AND create_date >= :cutoff_time
    """)

    result = await session.execute(query, {"cutoff_time": cutoff_time})
    clusters = result.fetchall()

    best_similarity = 0
    best_cluster_id = None

    for cluster in clusters:
        cluster_embedding = json.loads(cluster.embedding)
        similarity = cosine_similarity(message_embedding, cluster_embedding)

        if similarity > SIMILARITY_THRESHOLD and similarity > best_similarity:
            best_similarity = similarity
            best_cluster_id = cluster.id

    return best_cluster_id, best_similarity

async def create_cluster(session, message_id, message_text, embedding, forwarded_message_id=None):
    """Create a new cluster for a message with creation date."""
    query = text("""
    INSERT INTO clusters (text, embedding, create_date, forwarded_message_id)
    VALUES (:message_text, :embedding, CURRENT_TIMESTAMP, :forwarded_message_id)
    RETURNING id
    """)

    result = await session.execute(
        query,
        {
            "message_text": message_text,
            "embedding": json.dumps(embedding),
            "forwarded_message_id": forwarded_message_id
        }
    )
    cluster_id = result.fetchone()[0]

    # Link message to the new cluster
    await link_message_to_cluster(session, message_id, cluster_id)

    return cluster_id

async def link_message_to_cluster(session, message_id, cluster_id):
    """Link a message to a cluster."""
    query = text("""
    INSERT INTO cluster_messages (cluster_id, message_id)
    VALUES (:cluster_id, :message_id)
    """)

    await session.execute(
        query,
        {"cluster_id": cluster_id, "message_id": message_id}
    )

async def get_cluster_size(session, cluster_id):
    """Get the number of messages in a cluster."""
    query = text("""
    SELECT COUNT(*) as count FROM cluster_messages
    WHERE cluster_id = :cluster_id
    """)

    result = await session.execute(query, {"cluster_id": cluster_id})
    count = result.fetchone()[0]
    return count

async def get_cluster_messages(session, cluster_id):
    """Get all messages in a cluster."""
    query = text("""
    SELECT tm.id, tm.text, tm.channel_title, tm.message_id, tm.date
    FROM telegram_messages tm
    JOIN cluster_messages cm ON tm.id = cm.message_id
    WHERE cm.cluster_id = :cluster_id
    ORDER BY tm.date
    """)

    result = await session.execute(query, {"cluster_id": cluster_id})
    return result.fetchall()

async def generate_summary(messages):
    """Generate a summary of messages using a more affordable model."""
    messages_text = "\n\n---\n\n".join([msg.text for msg in messages])
    prompt = f"Please summarize these related news items in a concise paragraph using the language of the first message:\n\n{messages_text}"

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes related news items."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Failed to generate summary."

async def process_new_message(message):
    """Process a new message by calculating embedding, finding/creating cluster, and managing forwarding."""
    message_text = message.message
    if not message_text:
        return

    # Get embedding for the message
    embedding = await get_embedding(message_text)
    if not embedding:
        print("Failed to get embedding for message")
        return

    async with async_session_factory() as session:
        # Save message to database with embedding
        query = text("""
        INSERT INTO telegram_messages
        (message_id, date, text, channel_title, embedding)
        VALUES (:message_id, :date, :message_text, :channel_title, :embedding)
        RETURNING id
        """)

        result = await session.execute(
            query,
            {
                "message_id": message.id,
                "date": message.date,
                "message_text": message_text,
                "channel_title": message.chat.title if hasattr(message.chat, 'title') else "Unknown",
                "embedding": json.dumps(embedding)
            }
        )
        db_message_id = result.fetchone()[0]

        # Find similar cluster or create a new one
        cluster_id, similarity = await find_similar_cluster(session, embedding)

        # Forward message logic
        forwarded_message = None

        if cluster_id:
            # Message belongs to existing cluster
            await link_message_to_cluster(session, db_message_id, cluster_id)
            print(f"Message added to existing cluster {cluster_id} with similarity {similarity}")

            # Get the forwarded message ID from the cluster
            query = text("""
            SELECT forwarded_message_id FROM clusters
            WHERE id = :cluster_id
            """)
            result = await session.execute(query, {"cluster_id": cluster_id})
            cluster_info = result.fetchone()
    
            # Get the previous messages' channels in this cluster
            query = text("""
            SELECT DISTINCT tm.channel_title 
            FROM telegram_messages tm
            JOIN cluster_messages cm ON tm.id = cm.message_id
            WHERE cm.cluster_id = :cluster_id AND tm.id != :current_message_id
            """)
            result = await session.execute(query, {
                "cluster_id": cluster_id,
                "current_message_id": db_message_id
            })
            previous_channels = {row.channel_title for row in result.fetchall()}

            # If this message is from a new channel and we have a forwarded message
            current_channel = message.chat.title if hasattr(message.chat, 'title') else "Unknown"
            channel_username = message.chat.username if hasattr(message.chat, 'username') else None
            
            if current_channel not in previous_channels and cluster_info and cluster_info.forwarded_message_id:
                # Create a comment with a reference to the new message
                comment_text = f"📢 Similar message detected in "
            
                # Add direct message link if username is available
                if channel_username:
                    # Create a direct link to the message
                    message_link = f"https://t.me/{channel_username}/{message.id}"
                    comment_text += f"[{current_channel}]({message_link})"
                else:
                    comment_text += current_channel
            
                comment_text += f":\n\n{message_text[:150]}{'...' if len(message_text) > 150 else ''}"
            
                try:
                    await telegram_client.send_message(
                        entity=DESTINATION_CHANNEL,
                        message=comment_text,
                        comment_to=cluster_info.forwarded_message_id,
                        parse_mode='markdown'  # Enable markdown for the link
                    )
                    print(f"Added cross-channel reference comment for cluster {cluster_id}")
                except Exception as e:
                    print(f"Error posting cross-channel reference: {e}")
        else:
            # Create new cluster and forward the message
            try:
                forwarded_message = await telegram_client.forward_messages(
                    DESTINATION_CHANNEL,
                    message
                )
                forwarded_message_id = forwarded_message.id if forwarded_message else None
                print(f"Forwarded message to destination channel, ID: {forwarded_message_id}")
            except Exception as e:
                print(f"Error forwarding message: {e}")

            # Create new cluster with the forwarded message ID
            cluster_id = await create_cluster(session, db_message_id, message_text, embedding, forwarded_message_id)
            print(f"Created new cluster {cluster_id} for message")


        # # Check if this cluster now has 3 or more messages
        # cluster_size = await get_cluster_size(session, cluster_id)
        # 
        # if cluster_size == 3:
        #     # Get all messages in this cluster
        #     cluster_messages = await get_cluster_messages(session, cluster_id)
        # 
        #     # Generate summary
        #     summary = await generate_summary(cluster_messages)
        # 
        #     # Get the forwarded_message_id from the cluster
        #     query = text("""
        #     SELECT forwarded_message_id FROM clusters
        #     WHERE id = :cluster_id
        #     """)
        # 
        #     result = await session.execute(query, {"cluster_id": cluster_id})
        #     cluster_info = result.fetchone()
        # 
        #     if cluster_info and cluster_info.forwarded_message_id:
        #         try:
        #             comment = await telegram_client.send_message(
        #                 DESTINATION_CHANNEL,
        #                 summary,
        #                 comment_to=cluster_info.forwarded_message_id
        #             )
        #             print(f"Posted summary to cluster {cluster_id}, commenting on message {cluster_info.forwarded_message_id}")
        # 
        #             # Try to pin the comment
        #             try:
        #                 await telegram_client(functions.messages.UpdatePinnedMessageRequest(
        #                     peer=DESTINATION_CHANNEL,
        #                     id=comment.id,
        #                     pinned=True
        #                 ))
        #                 print(f"Successfully pinned summary comment for cluster {cluster_id}")
        #             except Exception as e:
        #                 print(f"Could not pin comment: {e}")
        #         except Exception as e:
        #             print(f"Error posting summary: {e}")
        #     else:
        #         print(f"No forwarded message ID found for cluster {cluster_id}")


        await session.commit()

async def listen_to_channels(channels):
    """Listen to specified channels for new messages."""
    @telegram_client.on(events.NewMessage(chats=channels))
    async def handle_new_message(event):
        await process_new_message(event.message)

    print(f"Now listening for new messages from channels: {', '.join(channels)}")

    # Keep the client running
    await telegram_client.run_until_disconnected()

async def subscribe_and_mute_channels(channels):
    """Subscribe to all channels in the list and mute them."""
    print(f"Attempting to subscribe and mute {len(channels)} channels...")

    for channel in channels:
        try:
            # Join the channel if not already joined
            try:
                channel_entity = await telegram_client.get_entity(channel)
                await telegram_client(functions.channels.JoinChannelRequest(channel=channel_entity))
                print(f"Subscribed to {channel}")
            except Exception as e:
                print(f"Could not subscribe to {channel}: {e}")
                continue

            # Mute the channel
            try:
                # Get the notification settings for the channel
                await telegram_client(functions.account.UpdateNotifySettingsRequest(
                    peer=channel_entity,
                    settings=types.InputPeerNotifySettings(
                        mute_until=2147483647,  # Mute forever (max timestamp)
                        show_previews=False,
                        silent=True
                    )
                ))
                print(f"Muted notifications for {channel}")
            except Exception as e:
                print(f"Could not mute {channel}: {e}")

        except Exception as e:
            print(f"Error processing {channel}: {e}")

    print("Channel subscription and muting complete")


async def main():
    """Main function to run the application."""
    # Connect to Telegram - make sure we're fully connected before setting up handlers
    print("Starting Telegram client...")
    await telegram_client.start()
    print("Starting bot client...")
    await bot.start(bot_token=BOT_TOKEN)

    # # Define channels to monitor (replace with actual channel names/IDs)
    channels_to_monitor = [
        '@gruntmedia',
        '@ssternenko',
        '@OstanniyCapitalist',
        '@torontotv'
    ]

    await subscribe_and_mute_channels(channels_to_monitor)

    # Define channels to monitor 
    # channels_to_monitor = [
    #     '@clustering1',
    #     '@clustering2',
    #     '@clustering3'
    # ]

    print(f"Setting up event handlers for: {channels_to_monitor}")
    # Register event handlers
    @telegram_client.on(events.NewMessage(chats=channels_to_monitor))
    async def handle_new_message(event):
        print(f"Received message from {event.chat.title}: {event.message.text[:50]}...")
        await process_new_message(event.message)

    print("Event handlers registered, waiting for messages...")
    # Keep the client running
    await telegram_client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())