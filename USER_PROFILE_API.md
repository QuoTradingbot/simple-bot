# User Profile API Endpoints

This document describes the user profile management endpoints added to the QuoTrading Flask API.

## Overview

The user profile endpoints allow authenticated users to view and update their own profile information. Authentication is done via license key, which can be provided either in the Authorization header or as a request parameter.

## Authentication

All user profile endpoints require authentication via license key:

**Method 1: Authorization Header (Recommended)**
```
Authorization: Bearer YOUR-LICENSE-KEY-HERE
```

**Method 2: Query Parameter (GET) or Request Body (PUT)**
```
?license_key=YOUR-LICENSE-KEY-HERE
```

## Endpoints

### GET /api/user/profile

Retrieve the authenticated user's profile information.

**Request:**
```bash
GET /api/user/profile
Authorization: Bearer YOUR-LICENSE-KEY-HERE
```

**Response (200 OK):**
```json
{
  "status": "success",
  "profile": {
    "account_id": "user123...",
    "email": "user@example.com",
    "license_key": "XXXX-....-YYYY",
    "license_type": "MONTHLY",
    "license_status": "ACTIVE",
    "license_expiration": "2025-01-15T00:00:00",
    "created_at": "2024-12-01T10:30:00",
    "updated_at": "2024-12-02T14:20:00",
    "last_heartbeat": "2024-12-02T15:00:00",
    "metadata": {
      "theme": "dark",
      "notifications": true
    },
    "whop_user_id": "whop_user_abc123",
    "whop_membership_id": "mem_xyz789"
  }
}
```

**Error Responses:**
- `401 Unauthorized` - License key missing or invalid
- `404 Not Found` - User profile not found
- `500 Internal Server Error` - Database or server error

### PUT /api/user/profile

Update the authenticated user's profile information.

**Updatable Fields:**
- `email` - User's email address (must be unique)
- `metadata` - JSON object containing user preferences and settings

**System-Managed Fields (Cannot be updated by user):**
- `account_id`, `license_key`, `license_type`, `license_status`, `license_expiration`, `whop_user_id`, `whop_membership_id`

**Request:**
```bash
PUT /api/user/profile
Authorization: Bearer YOUR-LICENSE-KEY-HERE
Content-Type: application/json

{
  "email": "newemail@example.com",
  "metadata": {
    "theme": "dark",
    "notifications": true,
    "timezone": "America/New_York"
  }
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Profile updated successfully",
  "profile": {
    "account_id": "user123...",
    "email": "newemail@example.com",
    "license_type": "MONTHLY",
    "license_status": "ACTIVE",
    "license_expiration": "2025-01-15T00:00:00",
    "updated_at": "2024-12-02T15:30:00",
    "metadata": {
      "theme": "dark",
      "notifications": true,
      "timezone": "America/New_York"
    }
  }
}
```

**Error Responses:**
- `400 Bad Request` - Invalid input (invalid email format, invalid metadata type, no fields to update)
- `401 Unauthorized` - License key missing or invalid
- `404 Not Found` - User profile not found
- `409 Conflict` - Email already in use by another account
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Database or server error

## Examples

### Python Example

```python
import requests

# Configuration
API_URL = "https://quotrading-flask-api.azurewebsites.net"
LICENSE_KEY = "YOUR-LICENSE-KEY-HERE"

# Get profile
headers = {"Authorization": f"Bearer {LICENSE_KEY}"}
response = requests.get(f"{API_URL}/api/user/profile", headers=headers)
profile = response.json()["profile"]
print(f"Email: {profile['email']}")

# Update profile
payload = {
    "metadata": {
        "theme": "dark",
        "notifications": True
    }
}
response = requests.put(
    f"{API_URL}/api/user/profile",
    headers=headers,
    json=payload
)
print(response.json()["message"])
```

### JavaScript Example

```javascript
const API_URL = "https://quotrading-flask-api.azurewebsites.net";
const LICENSE_KEY = "YOUR-LICENSE-KEY-HERE";

// Get profile
async function getProfile() {
  const response = await fetch(`${API_URL}/api/user/profile`, {
    headers: {
      "Authorization": `Bearer ${LICENSE_KEY}`
    }
  });
  const data = await response.json();
  console.log("Email:", data.profile.email);
}

// Update profile
async function updateProfile(email, metadata) {
  const response = await fetch(`${API_URL}/api/user/profile`, {
    method: "PUT",
    headers: {
      "Authorization": `Bearer ${LICENSE_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email, metadata })
  });
  const data = await response.json();
  console.log(data.message);
}

// Usage
getProfile();
updateProfile("newemail@example.com", { theme: "dark" });
```

### cURL Examples

**Get Profile:**
```bash
curl -X GET "https://quotrading-flask-api.azurewebsites.net/api/user/profile" \
  -H "Authorization: Bearer YOUR-LICENSE-KEY-HERE"
```

**Update Email:**
```bash
curl -X PUT "https://quotrading-flask-api.azurewebsites.net/api/user/profile" \
  -H "Authorization: Bearer YOUR-LICENSE-KEY-HERE" \
  -H "Content-Type: application/json" \
  -d '{"email": "newemail@example.com"}'
```

**Update Metadata:**
```bash
curl -X PUT "https://quotrading-flask-api.azurewebsites.net/api/user/profile" \
  -H "Authorization: Bearer YOUR-LICENSE-KEY-HERE" \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"theme": "dark", "notifications": true}}'
```

## Security Considerations

1. **Authentication Required**: All endpoints require a valid license key
2. **Rate Limiting**: Requests are rate-limited to prevent abuse (100 requests per minute per license)
3. **Partial License Key**: The license key in responses is partially masked for security
4. **Email Uniqueness**: Email addresses must be unique across all users
5. **HTTPS Only**: Always use HTTPS in production to protect license keys in transit
6. **Read-Only Fields**: System-managed fields cannot be modified by users

## Metadata Field

The `metadata` field is a flexible JSON object that can store any user preferences or settings. Common use cases:

```json
{
  "theme": "dark",
  "notifications": true,
  "timezone": "America/New_York",
  "language": "en",
  "trading_preferences": {
    "default_symbol": "ES",
    "risk_level": "medium"
  },
  "ui_settings": {
    "chart_type": "candlestick",
    "show_indicators": true
  }
}
```

## Best Practices

1. **Use Authorization Header**: Prefer the Authorization header over query parameters for better security
2. **Validate Input**: Always validate user input on the client side before sending to the API
3. **Handle Errors**: Implement proper error handling for all API responses
4. **Cache Profile Data**: Cache the profile data locally and only fetch when needed
5. **Update Incrementally**: Only send the fields you want to update, not the entire profile

## Rate Limits

- **Limit**: 100 requests per 60 seconds per license key
- **Scope**: Applies to the `/api/user/profile` endpoint
- **Response**: HTTP 429 Too Many Requests with error message
- **Reset**: Rate limit window resets after 60 seconds

## Support

For issues or questions about the user profile endpoints, contact:
- Email: support@quotrading.com
- Discord: QuoTrading Community
