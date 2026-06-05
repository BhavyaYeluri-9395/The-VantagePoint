def update_user_session(self, session_id, user_data):
    """Store or update user session information"""
    from datetime import datetime
    
    # First, try to update existing session
    result = self.user_sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "last_active": datetime.utcnow(),
                "user_agent": user_data.get('user_agent', 'Unknown'),
                "ip_address": user_data.get('ip_address', 'Unknown')
            },
            "$inc": {
                "total_predictions": 1
            },
            "$setOnInsert": {
                "session_id": session_id,
                "total_feedback": 0,
                "preferred_genres": [],
                "avg_rating_given": 0,
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )
    return result
