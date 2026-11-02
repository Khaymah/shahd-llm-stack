// Shopping cart: items are { sku, price, quantity }; prices are in cents.
export function addItem(cart, item) {
  const existing = cart.find((entry) => entry.sku === item.sku);
  if (existing) {
    existing.quantity += item.quantity;
    return cart;
  }
  return [...cart, { ...item }];
}

export function cartTotal(cart) {
  return cart.reduce((sum, entry) => sum + entry.price * entry.quantity, 0);
}

export function applyDiscount(totalCents, percent) {
  const clamped = Math.min(Math.max(percent, 0), 100);
  return Math.round(totalCents * (1 - clamped / 100));
}
