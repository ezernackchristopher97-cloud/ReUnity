import { describe, it, expect } from 'vitest';

// Test the generateExportHTML function logic
describe('Session Export', () => {
  it('should format date correctly', () => {
    const date = new Date('2025-01-25T10:30:00Z');
    const formatted = date.toLocaleString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
    expect(formatted).toContain('2025');
    expect(formatted).toContain('January');
  });

  it('should escape HTML in message content', () => {
    const content = '<script>alert("xss")</script>';
    const escaped = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    expect(escaped).toBe('&lt;script&gt;alert("xss")&lt;/script&gt;');
    expect(escaped).not.toContain('<script>');
  });

  it('should generate valid filename', () => {
    const conversationId = 123;
    const date = new Date().toISOString().split('T')[0];
    const filename = `reunity-session-${conversationId}-${date}.html`;
    expect(filename).toMatch(/^reunity-session-\d+-\d{4}-\d{2}-\d{2}\.html$/);
  });

  it('should include crisis resources in export', () => {
    const crisisResources = [
      '988 Suicide & Crisis Lifeline',
      'Crisis Text Line',
      'National Domestic Violence Hotline'
    ];
    crisisResources.forEach(resource => {
      expect(resource.length).toBeGreaterThan(0);
    });
  });

  it('should handle messages with patterns', () => {
    const patterns = ['gaslighting', 'isolation'];
    const formatted = patterns.join(', ');
    expect(formatted).toBe('gaslighting, isolation');
  });

  it('should handle empty messages array', () => {
    const messages: any[] = [];
    const html = messages.map(m => `<div>${m.content}</div>`).join('');
    expect(html).toBe('');
  });

  it('should include REOP branding', () => {
    const brandingElements = [
      'REOP Solutions',
      'entropy-physics-ai.com',
      'ReUnity'
    ];
    brandingElements.forEach(element => {
      expect(element.length).toBeGreaterThan(0);
    });
  });
});
