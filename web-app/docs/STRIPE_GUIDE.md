# ReUnity Stripe Integration Guide

**Complete Step-by-Step Instructions for Setting Up Payments (No Coding Experience Required)**

This guide walks you through setting up Stripe to accept payments in your ReUnity app. Whether you want to offer subscriptions, one-time donations, or premium features, this guide covers everything.

---

## Table of Contents

1. [Understanding Stripe](#part-1-understanding-stripe)
2. [Creating Your Stripe Account](#part-2-creating-your-stripe-account)
3. [Getting Your API Keys](#part-3-getting-your-api-keys)
4. [Adding Stripe to ReUnity](#part-4-adding-stripe-to-reunity)
5. [Setting Up Products and Prices](#part-5-setting-up-products-and-prices)
6. [Testing Payments](#part-6-testing-payments)
7. [Going Live](#part-7-going-live)
8. [Managing Payments](#part-8-managing-payments)

---

## Part 1: Understanding Stripe

### What is Stripe?

Stripe is a payment processing service that lets you accept credit cards, debit cards, and other payment methods. It handles all the complex security requirements so you don't have to.

### Stripe Fees

| Transaction Type | Fee (US) |
|------------------|----------|
| Credit/Debit Card | 2.9% + $0.30 per transaction |
| ACH Direct Debit | 0.8% (max $5) |
| International Cards | +1.5% |
| Currency Conversion | +1% |

**Example**: If someone pays $10, Stripe takes $0.59 (2.9% of $10 = $0.29, plus $0.30), and you receive $9.41.

### What You Can Do with Stripe

For ReUnity, you might want to:

1. **Subscriptions**: Monthly/yearly premium access ($9.99/month)
2. **One-time Donations**: Let users support the project
3. **Premium Features**: Unlock advanced features for a fee
4. **Pay-what-you-want**: Let users choose their amount

---

## Part 2: Creating Your Stripe Account

### Step 2.1: Sign Up for Stripe

1. Open your browser and go to [stripe.com](https://stripe.com).

2. Click **Start now** (or **Sign in** if you already have an account).

3. Enter your email address and create a password.

4. Click **Create account**.

5. Stripe will send a verification email. Open it and click **Verify email**.

### Step 2.2: Complete Your Business Profile

After verifying your email, Stripe asks for business information:

**Business Structure:**
- Select **Individual/Sole proprietor** (unless you have a registered business)

**Personal Information:**
- Legal name (as it appears on your ID)
- Date of birth
- Last 4 digits of SSN (required for US tax reporting)
- Home address

**Business Information:**
- Business name: REOP Solutions (or your business name)
- Business website: Your ReUnity web app URL
- Product description: "Mental health support application with AI companion"
- Industry: Software / Health & Wellness

**Bank Account for Payouts:**
- Routing number (9 digits, found on your checks or bank app)
- Account number

### Step 2.3: Verify Your Identity

Stripe may ask for identity verification:

1. Upload a photo of your government ID (driver's license or passport).
2. Take a selfie for verification.
3. Wait 1-2 business days for approval.

You can still use Stripe in test mode while waiting for verification.

---

## Part 3: Getting Your API Keys

API keys are like passwords that let your app talk to Stripe.

### Step 3.1: Access the Stripe Dashboard

1. Go to [dashboard.stripe.com](https://dashboard.stripe.com).
2. Sign in with your Stripe account.

### Step 3.2: Find Your API Keys

1. In the left sidebar, click **Developers**.
2. Click **API keys**.

You'll see two types of keys:

| Key Type | Starts With | Purpose |
|----------|-------------|---------|
| Publishable key | `pk_test_` or `pk_live_` | Used in frontend (safe to expose) |
| Secret key | `sk_test_` or `sk_live_` | Used in backend (NEVER expose) |

### Step 3.3: Copy Your Test Keys

For now, use the **test** keys (they have `_test_` in them):

1. Click **Reveal test key** next to the Secret key.
2. Copy both keys and save them somewhere safe (like a notes app).

**Important**: 
- `pk_test_...` = Publishable key (frontend)
- `sk_test_...` = Secret key (backend, keep private!)

### Step 3.4: Understanding Test vs Live Mode

| Mode | Purpose | Real Money? |
|------|---------|-------------|
| Test | Development and testing | No |
| Live | Real customers | Yes |

The toggle at the top of the Stripe dashboard switches between modes. Always develop in Test mode first!

---

## Part 4: Adding Stripe to ReUnity

### Step 4.1: Add Stripe Feature to Your Project

Open Terminal and navigate to your project:

```bash
cd ~/Projects/reunity-app
```

If you're using the Manus platform, Stripe integration is available as a feature. Otherwise, install Stripe manually:

```bash
pnpm add stripe @stripe/stripe-js
```

### Step 4.2: Add Environment Variables

Open your `.env` file:

```bash
nano .env
```

Add these lines (replace with your actual keys):

```
STRIPE_SECRET_KEY=sk_test_your_secret_key_here
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
```

Save and exit (Ctrl + X, Y, Enter).

### Step 4.3: Create Stripe Server File

Create a new file for Stripe backend logic:

```bash
nano server/stripe.ts
```

Add this code:

```typescript
import Stripe from 'stripe';

// Initialize Stripe with your secret key
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2023-10-16',
});

export { stripe };
```

Save and exit.

### Step 4.4: Create Checkout Endpoint

Open your routers file:

```bash
nano server/routers.ts
```

Add the Stripe import at the top:

```typescript
import { stripe } from './stripe';
```

Add a new procedure for creating checkout sessions:

```typescript
// Add this inside your router
createCheckoutSession: protectedProcedure
  .input(z.object({
    priceId: z.string(),
    successUrl: z.string(),
    cancelUrl: z.string(),
  }))
  .mutation(async ({ input, ctx }) => {
    const session = await stripe.checkout.sessions.create({
      customer_email: ctx.user.email,
      line_items: [
        {
          price: input.priceId,
          quantity: 1,
        },
      ],
      mode: 'subscription', // or 'payment' for one-time
      success_url: input.successUrl,
      cancel_url: input.cancelUrl,
      metadata: {
        userId: ctx.user.id,
      },
    });
    
    return { url: session.url };
  }),
```

Save and exit.

### Step 4.5: Create Frontend Payment Button

Create a new component for the payment button:

```bash
nano client/src/components/SubscribeButton.tsx
```

Add this code:

```typescript
import { Button } from '@/components/ui/button';
import { trpc } from '@/lib/trpc';

interface SubscribeButtonProps {
  priceId: string;
  children: React.ReactNode;
}

export function SubscribeButton({ priceId, children }: SubscribeButtonProps) {
  const createCheckout = trpc.createCheckoutSession.useMutation();

  const handleClick = async () => {
    const result = await createCheckout.mutateAsync({
      priceId,
      successUrl: `${window.location.origin}/success`,
      cancelUrl: `${window.location.origin}/pricing`,
    });
    
    if (result.url) {
      window.location.href = result.url;
    }
  };

  return (
    <Button 
      onClick={handleClick} 
      disabled={createCheckout.isPending}
    >
      {createCheckout.isPending ? 'Loading...' : children}
    </Button>
  );
}
```

Save and exit.

---

## Part 5: Setting Up Products and Prices

### Step 5.1: Create a Product in Stripe

1. Go to [dashboard.stripe.com](https://dashboard.stripe.com).

2. Make sure you're in **Test mode** (toggle at top).

3. In the left sidebar, click **Products**.

4. Click **Add product**.

5. Fill in the details:
   - **Name**: ReUnity Premium
   - **Description**: Unlimited access to all ReUnity features including advanced AI support, unlimited voice chat, and priority crisis response.
   - **Image**: Upload your app icon (optional)

6. Under **Pricing**:
   - Click **Add price**
   - **Pricing model**: Standard pricing
   - **Price**: $9.99
   - **Billing period**: Monthly (for subscriptions) or One time
   - **Currency**: USD

7. Click **Save product**.

### Step 5.2: Get the Price ID

After creating the product:

1. Click on the product you just created.
2. Under **Pricing**, you'll see your price listed.
3. Click on the price to expand it.
4. Copy the **Price ID** (starts with `price_`).

This is what you'll use in your code (the `priceId` parameter).

### Step 5.3: Create Multiple Pricing Tiers (Optional)

You might want different tiers:

| Tier | Price | Features |
|------|-------|----------|
| Basic | Free | Limited chat, basic grounding |
| Premium | $9.99/month | Unlimited chat, voice, all features |
| Supporter | $19.99/month | Everything + priority support |

Create each as a separate price on the same product, or create separate products.

---

## Part 6: Testing Payments

### Step 6.1: Use Test Card Numbers

Stripe provides test card numbers that work in test mode:

| Card Number | Result |
|-------------|--------|
| 4242 4242 4242 4242 | Successful payment |
| 4000 0000 0000 0002 | Card declined |
| 4000 0000 0000 9995 | Insufficient funds |
| 4000 0025 0000 3155 | Requires authentication (3D Secure) |

For all test cards:
- **Expiry**: Any future date (e.g., 12/34)
- **CVC**: Any 3 digits (e.g., 123)
- **ZIP**: Any 5 digits (e.g., 12345)

### Step 6.2: Test a Payment

1. Start your app locally:
   ```bash
   pnpm dev
   ```

2. Navigate to your pricing/subscribe page.

3. Click the subscribe button.

4. You'll be redirected to Stripe Checkout.

5. Enter the test card: `4242 4242 4242 4242`

6. Fill in any email, expiry (12/34), CVC (123), and name.

7. Click **Subscribe** or **Pay**.

8. You should be redirected to your success page.

### Step 6.3: Verify in Stripe Dashboard

1. Go to [dashboard.stripe.com](https://dashboard.stripe.com).

2. Click **Payments** in the left sidebar.

3. You should see your test payment listed.

4. Click on it to see details.

### Step 6.4: Test Webhooks (Advanced)

Webhooks notify your app when payment events happen (like successful payments or cancellations).

1. In Stripe Dashboard, go to **Developers** → **Webhooks**.

2. Click **Add endpoint**.

3. For local testing, use the Stripe CLI:
   ```bash
   brew install stripe/stripe-cli/stripe
   stripe login
   stripe listen --forward-to localhost:3000/api/stripe/webhook
   ```

4. Copy the webhook signing secret that appears.

5. Add to your `.env`:
   ```
   STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
   ```

---

## Part 7: Going Live

### Step 7.1: Complete Stripe Verification

Before accepting real payments:

1. Go to [dashboard.stripe.com/account](https://dashboard.stripe.com/account).

2. Complete any remaining verification steps.

3. Add your bank account for payouts.

4. Verify your identity if not already done.

### Step 7.2: Switch to Live Keys

1. In Stripe Dashboard, toggle from **Test mode** to **Live mode** (top of page).

2. Go to **Developers** → **API keys**.

3. Copy your live keys:
   - `pk_live_...` (Publishable)
   - `sk_live_...` (Secret)

### Step 7.3: Update Environment Variables

On your hosting platform (Railway, etc.):

1. Go to your project settings.

2. Update the environment variables:
   ```
   STRIPE_SECRET_KEY=sk_live_your_live_secret_key
   VITE_STRIPE_PUBLISHABLE_KEY=pk_live_your_live_publishable_key
   ```

3. If using webhooks, create a new webhook endpoint for your production URL and update:
   ```
   STRIPE_WEBHOOK_SECRET=whsec_your_live_webhook_secret
   ```

### Step 7.4: Create Live Products

Products created in test mode don't transfer to live mode. You need to recreate them:

1. Switch to **Live mode** in Stripe Dashboard.

2. Go to **Products** → **Add product**.

3. Create the same products/prices as in test mode.

4. Update your code with the new live Price IDs.

### Step 7.5: Deploy

Deploy your updated app:

```bash
railway up
```

Or redeploy through your hosting platform.

---

## Part 8: Managing Payments

### Viewing Payments

1. Go to [dashboard.stripe.com](https://dashboard.stripe.com).
2. Click **Payments** to see all transactions.
3. Click any payment for details, refund options, etc.

### Managing Subscriptions

1. Click **Subscriptions** in the left sidebar.
2. See all active, canceled, and past-due subscriptions.
3. Click any subscription to:
   - Cancel it
   - Change the plan
   - View payment history

### Issuing Refunds

1. Go to **Payments**.
2. Click on the payment to refund.
3. Click **Refund** in the top right.
4. Enter the amount (full or partial).
5. Click **Refund**.

### Payouts

Stripe automatically transfers your earnings to your bank account:

| Setting | Default |
|---------|---------|
| Payout schedule | Daily (2-day rolling) |
| Minimum payout | $0.01 |
| Payout speed | 2 business days |

To change payout settings:
1. Go to **Settings** → **Payouts**.
2. Adjust schedule (daily, weekly, monthly).

### Viewing Reports

1. Go to **Reports** in the left sidebar.
2. View:
   - Revenue over time
   - Successful vs failed payments
   - Subscription metrics
   - Payout history

---

## Pricing Page Example

Here's a simple pricing page you can add to ReUnity:

Create `client/src/pages/Pricing.tsx`:

```typescript
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Check } from 'lucide-react';
import { SubscribeButton } from '@/components/SubscribeButton';

const plans = [
  {
    name: 'Free',
    price: '$0',
    description: 'Basic mental health support',
    features: [
      '5 AI chat messages per day',
      'Basic grounding techniques',
      'Crisis resources',
      'Mood tracking',
    ],
    priceId: null,
    buttonText: 'Current Plan',
  },
  {
    name: 'Premium',
    price: '$9.99',
    period: '/month',
    description: 'Full access to all features',
    features: [
      'Unlimited AI chat',
      'Voice conversations',
      'All grounding techniques',
      'Advanced mood analytics',
      'Symptom tracking',
      'Guided meditations',
      'Priority support',
    ],
    priceId: 'price_YOUR_PRICE_ID_HERE', // Replace with your actual price ID
    buttonText: 'Subscribe',
    popular: true,
  },
  {
    name: 'Supporter',
    price: '$19.99',
    period: '/month',
    description: 'Support our mission',
    features: [
      'Everything in Premium',
      'Early access to new features',
      'Supporter badge',
      'Help us help others',
    ],
    priceId: 'price_YOUR_SUPPORTER_PRICE_ID', // Replace with your actual price ID
    buttonText: 'Support Us',
  },
];

export default function Pricing() {
  return (
    <div className="container py-12">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Choose Your Plan</h1>
        <p className="text-muted-foreground text-lg">
          Support your mental health journey with the right plan for you
        </p>
      </div>
      
      <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {plans.map((plan) => (
          <Card 
            key={plan.name}
            className={plan.popular ? 'border-primary shadow-lg scale-105' : ''}
          >
            {plan.popular && (
              <div className="bg-primary text-primary-foreground text-center py-1 text-sm font-medium">
                Most Popular
              </div>
            )}
            <CardHeader>
              <CardTitle>{plan.name}</CardTitle>
              <CardDescription>{plan.description}</CardDescription>
              <div className="mt-4">
                <span className="text-4xl font-bold">{plan.price}</span>
                {plan.period && (
                  <span className="text-muted-foreground">{plan.period}</span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-2">
                    <Check className="h-4 w-4 text-primary" />
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
            <CardFooter>
              {plan.priceId ? (
                <SubscribeButton priceId={plan.priceId} className="w-full">
                  {plan.buttonText}
                </SubscribeButton>
              ) : (
                <Button variant="outline" className="w-full" disabled>
                  {plan.buttonText}
                </Button>
              )}
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
```

---

## Handling Subscription Status

To check if a user has an active subscription:

### Add to Database Schema

In `drizzle/schema.ts`, add subscription fields to your users table:

```typescript
export const users = sqliteTable('users', {
  // ... existing fields
  stripeCustomerId: text('stripe_customer_id'),
  subscriptionStatus: text('subscription_status'), // 'active', 'canceled', 'past_due', null
  subscriptionId: text('subscription_id'),
  subscriptionEndDate: integer('subscription_end_date'),
});
```

Run migration:

```bash
pnpm db:push
```

### Create Webhook Handler

Webhooks update your database when subscription status changes:

```typescript
// server/stripe-webhook.ts
import { stripe } from './stripe';
import { db } from './db';
import { users } from '../drizzle/schema';
import { eq } from 'drizzle-orm';

export async function handleStripeWebhook(body: string, signature: string) {
  const event = stripe.webhooks.constructEvent(
    body,
    signature,
    process.env.STRIPE_WEBHOOK_SECRET!
  );

  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object;
      const userId = session.metadata?.userId;
      
      if (userId) {
        await db.update(users)
          .set({
            stripeCustomerId: session.customer as string,
            subscriptionId: session.subscription as string,
            subscriptionStatus: 'active',
          })
          .where(eq(users.id, userId));
      }
      break;
    }
    
    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      const subscription = event.data.object;
      
      await db.update(users)
        .set({
          subscriptionStatus: subscription.status,
          subscriptionEndDate: subscription.current_period_end,
        })
        .where(eq(users.stripeCustomerId, subscription.customer as string));
      break;
    }
  }

  return { received: true };
}
```

---

## Security Checklist

Before going live, verify:

| Item | Status |
|------|--------|
| Secret key is ONLY in backend code | ☐ |
| Secret key is in environment variables, not hardcoded | ☐ |
| Webhook signature is verified | ☐ |
| HTTPS is enabled on your site | ☐ |
| Test mode is OFF in production | ☐ |
| Live keys are used in production | ☐ |

---

## Troubleshooting

### "No such price" Error
- Make sure you're using the correct Price ID.
- Verify you're in the right mode (test vs live).
- Price IDs are different between test and live mode.

### Webhook Not Receiving Events
- Check the webhook URL is correct.
- Verify the webhook secret matches.
- Check Stripe Dashboard → Developers → Webhooks → Event logs.

### Payment Declined
- In test mode, use the correct test card numbers.
- In live mode, the customer's card may have issues.

### Checkout Page Not Loading
- Verify your publishable key is correct.
- Check browser console for errors.
- Ensure CORS is configured if using a custom domain.

---

## Costs Summary

| Item | Cost |
|------|------|
| Stripe Account | Free |
| Per Transaction | 2.9% + $0.30 |
| Monthly Fee | None |
| Chargebacks | $15 per dispute |

---

## Quick Reference

| Action | Where |
|--------|-------|
| View payments | dashboard.stripe.com → Payments |
| Create products | dashboard.stripe.com → Products |
| Get API keys | dashboard.stripe.com → Developers → API keys |
| View webhooks | dashboard.stripe.com → Developers → Webhooks |
| Issue refund | Click payment → Refund |
| Cancel subscription | Click subscription → Cancel |

---

## Additional Resources

For more detailed information:

- Stripe Documentation: [stripe.com/docs](https://stripe.com/docs)
- Stripe API Reference: [stripe.com/docs/api](https://stripe.com/docs/api)
- Stripe Support: [support.stripe.com](https://support.stripe.com)

---

**Congratulations!** You now have payment processing set up for ReUnity! 💳🎉
