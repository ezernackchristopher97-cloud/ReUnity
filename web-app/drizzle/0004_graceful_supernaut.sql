CREATE TABLE `clientEntropySnapshots` (
	`id` int AUTO_INCREMENT NOT NULL,
	`relationshipId` int NOT NULL,
	`clientId` int NOT NULL,
	`snapshotDate` timestamp NOT NULL,
	`avgEntropyScore` varchar(10),
	`entropyTrend` varchar(20),
	`dominantStates` json,
	`detectedPatterns` json,
	`journalEntryCount` int DEFAULT 0,
	`checkInsCompleted` int DEFAULT 0,
	`checkInsMissed` int DEFAULT 0,
	`crisisEventsCount` int DEFAULT 0,
	`aiSummary` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `clientEntropySnapshots_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `therapistAlerts` (
	`id` int AUTO_INCREMENT NOT NULL,
	`relationshipId` int NOT NULL,
	`therapistId` int NOT NULL,
	`clientId` int NOT NULL,
	`alertType` enum('crisis','high_entropy','missed_checkin','concerning_pattern','progress') NOT NULL,
	`severity` enum('low','medium','high','critical') NOT NULL DEFAULT 'medium',
	`title` varchar(255) NOT NULL,
	`description` text NOT NULL,
	`relatedData` json,
	`isViewed` boolean DEFAULT false,
	`viewedAt` timestamp,
	`isAcknowledged` boolean DEFAULT false,
	`acknowledgmentNotes` text,
	`acknowledgedAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `therapistAlerts_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `therapistClientRelationships` (
	`id` int AUTO_INCREMENT NOT NULL,
	`therapistId` int NOT NULL,
	`clientId` int NOT NULL,
	`status` enum('pending','active','paused','ended') NOT NULL DEFAULT 'pending',
	`consentedAt` timestamp,
	`consentedDataTypes` json,
	`crisisAlertsEnabled` boolean DEFAULT true,
	`therapistNotes` text,
	`endedAt` timestamp,
	`endReason` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `therapistClientRelationships_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `therapistProfiles` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`licenseNumber` varchar(100) NOT NULL,
	`licenseState` varchar(100) NOT NULL,
	`licenseType` varchar(50) NOT NULL,
	`specializations` json,
	`practiceName` varchar(255),
	`phone` varchar(20),
	`isVerified` boolean DEFAULT false,
	`verifiedAt` timestamp,
	`acceptingClients` boolean DEFAULT true,
	`bio` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `therapistProfiles_id` PRIMARY KEY(`id`),
	CONSTRAINT `therapistProfiles_userId_unique` UNIQUE(`userId`)
);
