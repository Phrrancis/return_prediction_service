(async function () {
  const cart = await fetch('/cart.js').then(res => res.json());

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      user_id: "guest",
      cart: cart.items.map(i => ({
        product_id: i.product_id,
        price: i.price / 100,
        category: i.product_type,
        size: i.variant_title
      }))
    })
  });

  const data = await response.json();

  if (data.action === "add_shipping_fee") {
    alert("⚠️ High return risk: shipping fee may apply");
  }

  if (data.action === "show_warning") {
    alert("⚠️ Items in your cart may not fit well");
  }
})();