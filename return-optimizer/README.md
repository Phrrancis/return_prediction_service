# Return Optimizer MVP

A lightweight FastAPI prototype that estimates return risk from cart-level product data and returns a simple risk score plus recommended action.

## Features

- FastAPI backend
- Request validation with Pydantic
- Rule-based return risk scoring
- Synthetic data generator for experimentation
- Basic health check endpoint
- Unit tests for model logic and API routes

## Run Tests

- Python 3.10+ recommended
- pip

## Shopify Script

The `shopify-app/inject.js` script demonstrates how a storefront could send cart data to the backend prediction endpoint.

## Notes

- This project uses a rule-based scoring approach, not a trained ML model.
- Thresholds and scoring logic can be tuned as the project evolves.
- Validation is enforced through the request schemas to reduce invalid input.

## Next Improvements

- Add a proper response model
- Add logging and metrics
- Add environment-based configuration
- Expand the feature set
- Replace rules with a trained model if needed
