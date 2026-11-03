// Thin client for the shop backend.
const BASE_URL = "https://api.cornershop.example/v1";

export async function fetchProduct(id) {
  const response = await fetch(`${BASE_URL}/products/${id}`);
  if (!response.ok) throw new Error(`product ${id}: HTTP ${response.status}`);
  return response.json();
}

export async function placeOrder(cart) {
  const response = await fetch(`${BASE_URL}/orders`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ items: cart }),
  });
  if (!response.ok) throw new Error(`order failed: HTTP ${response.status}`);
  return response.json();
}
