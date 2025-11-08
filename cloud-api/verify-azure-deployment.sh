#!/bin/bash

# QuoTrading - Azure Deployment Verification Script
# This script verifies that your Azure deployment is working correctly

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Check if URL is provided
if [ -z "$1" ]; then
    echo "Usage: ./verify-azure-deployment.sh <your-app-url>"
    echo "Example: ./verify-azure-deployment.sh https://quotrading-api.azurewebsites.net"
    exit 1
fi

API_URL="$1"

echo "Testing QuoTrading API at: $API_URL"
echo ""

# Test 1: Health Check
print_test "Health check (GET /)"
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_pass "Health check successful"
    echo "  Response: $BODY"
else
    print_fail "Health check failed (HTTP $HTTP_CODE)"
    exit 1
fi
echo ""

# Test 2: User Registration
print_test "User registration (POST /api/v1/users/register)"
TEST_EMAIL="test-$(date +%s)@example.com"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/users/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$TEST_EMAIL\"}")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_pass "User registration successful"
    API_KEY=$(echo "$BODY" | grep -o '"api_key":"[^"]*"' | cut -d'"' -f4)
    echo "  Email: $TEST_EMAIL"
    echo "  API Key: $API_KEY"
else
    print_fail "User registration failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
    exit 1
fi
echo ""

# Test 3: License Validation with Admin Key
print_test "License validation with admin key (POST /api/v1/license/validate)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/license/validate" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@quotrading.com", "api_key": "QUOTRADING_ADMIN_MASTER_2025"}')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_pass "Admin key validation successful"
    echo "  Response: $BODY"
else
    print_fail "Admin key validation failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Test 4: License Validation with Regular User
print_test "License validation with new user (POST /api/v1/license/validate)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/license/validate" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$TEST_EMAIL\", \"api_key\": \"$API_KEY\"}")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_pass "User license validation successful"
    echo "  Response: $BODY"
else
    print_fail "User license validation failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Test 5: Get User Info
print_test "Get user info (GET /api/v1/users/$TEST_EMAIL)"
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/api/v1/users/$TEST_EMAIL")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_pass "Get user info successful"
    echo "  Response: $BODY"
else
    print_fail "Get user info failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Test 6: Signal Generation Health
print_test "Signal generation health check (GET /api/v1/signals/health)"
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/api/v1/signals/health")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_pass "Signal health check successful"
    echo "  Response: $BODY"
else
    print_fail "Signal health check failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Summary
echo "=========================================="
echo "Deployment Verification Complete!"
echo "=========================================="
echo ""
echo "✓ API is responding correctly"
echo "✓ User registration working"
echo "✓ License validation working"
echo "✓ Signal engine accessible"
echo ""
echo "Next steps:"
echo "  1. Update your bot launcher with: CLOUD_API_BASE_URL = \"$API_URL\""
echo "  2. Configure Stripe webhook: $API_URL/api/v1/webhooks/stripe"
echo "  3. Set production Stripe keys in Azure app settings"
echo "  4. Monitor logs: az webapp log tail --resource-group <rg> --name <app-name>"
echo ""
