/**
 * Tests for Belief Systems Module
 * All beliefs are treated with equal respect and dignity
 */

import { describe, it, expect } from 'vitest';
import {
  getBeliefSystem,
  getBeliefsByCategory,
  getComfortingPhrase,
  getCopingStrategies,
  getCrisisSupport,
  searchBeliefSystems,
  getAllBeliefIds,
  getResponseGuidance,
  getUniversalComfort,
  beliefSystems
} from './belief-systems';

describe('Belief Systems Module', () => {
  describe('getBeliefSystem', () => {
    it('should return Christianity belief system', () => {
      const belief = getBeliefSystem('christianity');
      expect(belief).toBeDefined();
      expect(belief?.name).toBe('Christianity');
      expect(belief?.category).toBe('religious');
      expect(belief?.coreBeliefs.length).toBeGreaterThan(0);
    });

    it('should return Islam belief system', () => {
      const belief = getBeliefSystem('islam');
      expect(belief).toBeDefined();
      expect(belief?.name).toBe('Islam');
      expect(belief?.comfortingPhrases.some(p => p.includes('Quran'))).toBe(true);
    });

    it('should return Judaism belief system', () => {
      const belief = getBeliefSystem('judaism');
      expect(belief).toBeDefined();
      expect(belief?.name).toBe('Judaism');
    });

    it('should return Buddhism belief system', () => {
      const belief = getBeliefSystem('buddhism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('eastern');
    });

    it('should return Hinduism belief system', () => {
      const belief = getBeliefSystem('hinduism');
      expect(belief).toBeDefined();
      expect(belief?.sacredTexts).toContain('Bhagavad Gita');
    });

    it('should return Sikhism belief system', () => {
      const belief = getBeliefSystem('sikhism');
      expect(belief).toBeDefined();
      expect(belief?.practices).toContain('Langar');
    });

    it('should return Taoism belief system', () => {
      const belief = getBeliefSystem('taoism');
      expect(belief).toBeDefined();
      expect(belief?.sacredTexts).toContain('Tao Te Ching');
    });

    it('should return Existentialism belief system', () => {
      const belief = getBeliefSystem('existentialism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('philosophical');
    });

    it('should return Stoicism belief system', () => {
      const belief = getBeliefSystem('stoicism');
      expect(belief).toBeDefined();
      expect(belief?.comfortingPhrases.some(p => p.includes('Marcus Aurelius'))).toBe(true);
    });

    it('should return Nihilism belief system', () => {
      const belief = getBeliefSystem('nihilism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('philosophical');
    });

    it('should return Absurdism belief system', () => {
      const belief = getBeliefSystem('absurdism');
      expect(belief).toBeDefined();
      expect(belief?.comfortingPhrases.some(p => p.includes('Camus'))).toBe(true);
    });

    it('should return Solipsism belief system', () => {
      const belief = getBeliefSystem('solipsism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('philosophical');
    });

    it('should return Atheism belief system', () => {
      const belief = getBeliefSystem('atheism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('secular');
    });

    it('should return Agnosticism belief system', () => {
      const belief = getBeliefSystem('agnosticism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('secular');
    });

    it('should return Secular Humanism belief system', () => {
      const belief = getBeliefSystem('secular-humanism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('secular');
    });

    it('should return Paganism belief system', () => {
      const belief = getBeliefSystem('paganism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('spiritual');
    });

    it('should return Wicca belief system', () => {
      const belief = getBeliefSystem('wicca');
      expect(belief).toBeDefined();
      expect(belief?.practices).toContain('Circle casting');
    });

    it('should return New Age belief system', () => {
      const belief = getBeliefSystem('new-age');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('spiritual');
    });

    it('should return Shamanism belief system', () => {
      const belief = getBeliefSystem('shamanism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('indigenous');
    });

    it('should return Animism belief system', () => {
      const belief = getBeliefSystem('animism');
      expect(belief).toBeDefined();
      expect(belief?.category).toBe('indigenous');
    });

    it('should return undefined for unknown belief', () => {
      const belief = getBeliefSystem('unknown-belief');
      expect(belief).toBeUndefined();
    });

    it('should be case insensitive', () => {
      const belief = getBeliefSystem('CHRISTIANITY');
      expect(belief).toBeDefined();
      expect(belief?.name).toBe('Christianity');
    });
  });

  describe('getBeliefsByCategory', () => {
    it('should return all religious beliefs', () => {
      const beliefs = getBeliefsByCategory('religious');
      expect(beliefs.length).toBeGreaterThan(0);
      expect(beliefs.every(b => b.category === 'religious')).toBe(true);
    });

    it('should return all eastern beliefs', () => {
      const beliefs = getBeliefsByCategory('eastern');
      expect(beliefs.length).toBeGreaterThan(0);
      expect(beliefs.some(b => b.id === 'buddhism')).toBe(true);
    });

    it('should return all philosophical beliefs', () => {
      const beliefs = getBeliefsByCategory('philosophical');
      expect(beliefs.length).toBeGreaterThan(0);
      expect(beliefs.some(b => b.id === 'existentialism')).toBe(true);
    });

    it('should return all secular beliefs', () => {
      const beliefs = getBeliefsByCategory('secular');
      expect(beliefs.length).toBeGreaterThan(0);
      expect(beliefs.some(b => b.id === 'atheism')).toBe(true);
    });

    it('should return all spiritual beliefs', () => {
      const beliefs = getBeliefsByCategory('spiritual');
      expect(beliefs.length).toBeGreaterThan(0);
    });

    it('should return all indigenous beliefs', () => {
      const beliefs = getBeliefsByCategory('indigenous');
      expect(beliefs.length).toBeGreaterThan(0);
    });
  });

  describe('getComfortingPhrase', () => {
    it('should return a comforting phrase for Christianity', () => {
      const phrase = getComfortingPhrase('christianity');
      expect(phrase).toBeDefined();
      expect(typeof phrase).toBe('string');
    });

    it('should return a comforting phrase for Buddhism', () => {
      const phrase = getComfortingPhrase('buddhism');
      expect(phrase).toBeDefined();
    });

    it('should return a comforting phrase for Stoicism', () => {
      const phrase = getComfortingPhrase('stoicism');
      expect(phrase).toBeDefined();
    });

    it('should return undefined for unknown belief', () => {
      const phrase = getComfortingPhrase('unknown');
      expect(phrase).toBeUndefined();
    });
  });

  describe('getCopingStrategies', () => {
    it('should return coping strategies for Islam', () => {
      const strategies = getCopingStrategies('islam');
      expect(strategies.length).toBeGreaterThan(0);
      expect(strategies.some(s => s.includes('Salah') || s.includes('prayer'))).toBe(true);
    });

    it('should return coping strategies for Buddhism', () => {
      const strategies = getCopingStrategies('buddhism');
      expect(strategies.length).toBeGreaterThan(0);
      expect(strategies.some(s => s.toLowerCase().includes('meditation'))).toBe(true);
    });

    it('should return coping strategies for Atheism', () => {
      const strategies = getCopingStrategies('atheism');
      expect(strategies.length).toBeGreaterThan(0);
    });

    it('should return empty array for unknown belief', () => {
      const strategies = getCopingStrategies('unknown');
      expect(strategies).toEqual([]);
    });
  });

  describe('getCrisisSupport', () => {
    it('should return crisis support for Christianity', () => {
      const support = getCrisisSupport('christianity');
      expect(support.length).toBeGreaterThan(0);
    });

    it('should return crisis support for Nihilism', () => {
      const support = getCrisisSupport('nihilism');
      expect(support.length).toBeGreaterThan(0);
    });

    it('should return empty array for unknown belief', () => {
      const support = getCrisisSupport('unknown');
      expect(support).toEqual([]);
    });
  });

  describe('searchBeliefSystems', () => {
    it('should find beliefs by name', () => {
      const results = searchBeliefSystems('buddhism');
      expect(results.length).toBeGreaterThan(0);
      expect(results.some(b => b.id === 'buddhism')).toBe(true);
    });

    it('should find beliefs by description keywords', () => {
      const results = searchBeliefSystems('enlightenment');
      expect(results.length).toBeGreaterThan(0);
    });

    it('should find beliefs by core belief keywords', () => {
      const results = searchBeliefSystems('meaning');
      expect(results.length).toBeGreaterThan(0);
    });

    it('should return empty array for no matches', () => {
      const results = searchBeliefSystems('xyznonexistent');
      expect(results).toEqual([]);
    });
  });

  describe('getAllBeliefIds', () => {
    it('should return all belief system IDs', () => {
      const ids = getAllBeliefIds();
      expect(ids.length).toBeGreaterThan(20);
      expect(ids).toContain('christianity');
      expect(ids).toContain('buddhism');
      expect(ids).toContain('atheism');
      expect(ids).toContain('existentialism');
    });
  });

  describe('getResponseGuidance', () => {
    it('should return guidance for Christianity', () => {
      const guidance = getResponseGuidance('christianity');
      expect(guidance).toContain('God');
    });

    it('should return guidance for Islam', () => {
      const guidance = getResponseGuidance('islam');
      expect(guidance).toContain('Islamic');
    });

    it('should return guidance for Atheism', () => {
      const guidance = getResponseGuidance('atheism');
      expect(guidance).toContain('secular');
    });

    it('should return guidance for Stoicism', () => {
      const guidance = getResponseGuidance('stoicism');
      expect(guidance).toContain('Stoic');
    });

    it('should return empty string for unknown belief', () => {
      const guidance = getResponseGuidance('unknown');
      expect(guidance).toBe('');
    });
  });

  describe('getUniversalComfort', () => {
    it('should return a universal comforting message', () => {
      const message = getUniversalComfort();
      expect(message).toBeDefined();
      expect(typeof message).toBe('string');
      expect(message.length).toBeGreaterThan(0);
    });

    it('should return different messages on multiple calls', () => {
      const messages = new Set();
      for (let i = 0; i < 20; i++) {
        messages.add(getUniversalComfort());
      }
      // Should have some variety
      expect(messages.size).toBeGreaterThan(1);
    });
  });

  describe('Belief System Completeness', () => {
    it('should have all required fields for each belief system', () => {
      for (const belief of Object.values(beliefSystems)) {
        expect(belief.id).toBeDefined();
        expect(belief.name).toBeDefined();
        expect(belief.category).toBeDefined();
        expect(belief.description).toBeDefined();
        expect(belief.coreBeliefs.length).toBeGreaterThan(0);
        expect(belief.copingStrategies.length).toBeGreaterThan(0);
        expect(belief.comfortingPhrases.length).toBeGreaterThan(0);
      }
    });

    it('should have crisis support for each belief system', () => {
      for (const belief of Object.values(beliefSystems)) {
        expect(belief.crisisSupport).toBeDefined();
        expect(belief.crisisSupport!.length).toBeGreaterThan(0);
      }
    });
  });

  describe('All Beliefs Welcome', () => {
    it('should include major world religions', () => {
      expect(getBeliefSystem('christianity')).toBeDefined();
      expect(getBeliefSystem('islam')).toBeDefined();
      expect(getBeliefSystem('judaism')).toBeDefined();
      expect(getBeliefSystem('hinduism')).toBeDefined();
      expect(getBeliefSystem('buddhism')).toBeDefined();
      expect(getBeliefSystem('sikhism')).toBeDefined();
    });

    it('should include philosophical frameworks', () => {
      expect(getBeliefSystem('existentialism')).toBeDefined();
      expect(getBeliefSystem('stoicism')).toBeDefined();
      expect(getBeliefSystem('nihilism')).toBeDefined();
      expect(getBeliefSystem('absurdism')).toBeDefined();
      expect(getBeliefSystem('solipsism')).toBeDefined();
    });

    it('should include secular perspectives', () => {
      expect(getBeliefSystem('atheism')).toBeDefined();
      expect(getBeliefSystem('agnosticism')).toBeDefined();
      expect(getBeliefSystem('secular-humanism')).toBeDefined();
    });

    it('should include spiritual traditions', () => {
      expect(getBeliefSystem('paganism')).toBeDefined();
      expect(getBeliefSystem('wicca')).toBeDefined();
      expect(getBeliefSystem('new-age')).toBeDefined();
      expect(getBeliefSystem('shamanism')).toBeDefined();
      expect(getBeliefSystem('animism')).toBeDefined();
    });

    it('should include eastern philosophies', () => {
      expect(getBeliefSystem('taoism')).toBeDefined();
      expect(getBeliefSystem('confucianism')).toBeDefined();
    });
  });
});
