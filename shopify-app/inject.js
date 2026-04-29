(async function () {
  // Set RETURN_OPTIMIZER_URL and RETURN_OPTIMIZER_API_KEY via your Shopify theme settings
  // or replace the defaults below before deploying.
  const API_URL = window.RETURN_OPTIMIZER_URL || "https://your-api-domain.com";
  const API_KEY = window.RETURN_OPTIMIZER_API_KEY || "";

  const cart = await fetch("/cart.js").then((res) => res.json());

  let response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({
        user_id: window.__st?.cid || "guest",
        cart: cart.items.map((i) => ({
          product_id: String(i.product_id),
          price: i.price / 100,
          category: (i.product_type || "other").toLowerCase(),
          size: i.variant_title || "one-size",
        })),
      }),
    });
  } catch (err) {
    console.error("[ReturnOptimizer] Network error:", err);
    return;
  }

  if (!response.ok) {
    console.error("[ReturnOptimizer] API error:", response.status);
    return;
  }

  const data = await response.json();

  if (data.action === "add_shipping_fee") {
    alert("⚠️ High return risk: a shipping fee may apply to this order.");
  } else if (data.action === "show_warning") {
    alert("⚠️ Some items in your cart have a higher chance of being returned.");
  }
})();
