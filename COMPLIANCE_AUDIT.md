# ReUnity Compliance Audit

## Directive Items

### 1. Global Disclaimer — REQUIRED
- [ ] Onboarding screen
- [ ] Settings page
- [ ] Footer of web app
- [ ] App Store description metadata

Exact text: "ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services."

### 2. Legal Links (Required for App Store)
- [ ] Privacy Policy page created
- [ ] Terms of Service page created
- [ ] Accessible from settings
- [ ] Linked in app footer
- [ ] Hosted on reunityai.com
- [ ] Added to App Store submission metadata

### 3. Remove Strong Medical Claims
- [ ] Search entire project for diagnosis/treatment/clinical claims
- [ ] Replace with support/educational/informational/reflective language

### 4. Entropy Engine Validation
- [ ] Shannon entropy calculation intact
- [ ] Jensen-Shannon divergence intact
- [ ] State classification thresholds intact
- [ ] Protective pattern detection intact
- [ ] No placeholder math
- [ ] No random outputs
- [ ] No mocked entropy values

### 5. Data Handling + Storage
- [ ] No plaintext storage of user entries
- [ ] Local-first design verified
- [ ] Encryption for stored user data
- [ ] No silent analytics tracking / opt-in required

### 6. Build Process
- [ ] Remove console logs (production)
- [ ] Remove dev warnings (production)
- [ ] Confirm HTTPS enforced

### 7. App Store Prep Checklist
- [ ] App icon 1024x1024
- [ ] Screenshots guidance
- [ ] Age rating questionnaire guidance
- [ ] Privacy questionnaire guidance
- [ ] Data usage categories declared
- [ ] Encryption disclosure guidance

### 8. Sync Mobile + Web Consistency
- [ ] Same disclaimer text
- [ ] Same terminology
- [ ] Same entropy logic
- [ ] Same state naming

### 9. Stripe Removal (Beta)
- [x] Removed Stripe imports from routers.ts
- [x] Removed subscription router from routers.ts
- [x] Removed Stripe webhook handler from index.ts
- [x] Simplified updateSubscription to always return free tier
- [x] Server running clean with 0 TypeScript errors
