import { test, expect } from '@playwright/test';

test('Chatbot responds correctly', async ({ page }) => {
  await page.goto('http://localhost:8501');
  await page.fill('input[type="text"]', 'What is the stock price of AAPL?');
  await page.keyboard.press('Enter');
  await page.wait_for_selector('text=AAPL current price');
});
